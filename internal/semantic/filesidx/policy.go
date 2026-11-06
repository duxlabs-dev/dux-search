package filesidx

import (
	"bufio"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/godbus/dbus/v5"
)

// Pause reasons reported by Progress.PausedReason.
const (
	PauseBattery          = "battery"           // running on battery power
	PauseUserActive       = "user_active"       // the desktop session is not idle
	PauseModelUnavailable = "model_unavailable" // the embedding model is not installed
)

// PowerSource tells whether the machine runs on battery.
type PowerSource interface {
	OnBattery(ctx context.Context) (bool, error)
}

// ActivitySource tells whether the user is working at the machine.
type ActivitySource interface {
	// UserActive reports activity; known is false when the system cannot tell.
	UserActive(ctx context.Context) (active, known bool, err error)
}

const (
	upowerDest   = "org.freedesktop.UPower"
	upowerPath   = "/org/freedesktop/UPower"
	logindDest   = "org.freedesktop.login1"
	logindUser   = "/org/freedesktop/login1/user/self"
	propsGet     = "org.freedesktop.DBus.Properties.Get"
	dbusCallWait = 5 * time.Second
)

// systemBus lazily keeps one system bus connection.
type systemBus struct{ conn *dbus.Conn }

func (b *systemBus) get() (*dbus.Conn, error) {
	if b.conn != nil && b.conn.Connected() {
		return b.conn, nil
	}
	if b.conn != nil {
		b.conn.Close()
		b.conn = nil
	}
	c, err := dbus.ConnectSystemBus()
	if err != nil {
		return nil, err
	}
	b.conn = c
	return c, nil
}

func (b *systemBus) close() {
	if b.conn != nil {
		b.conn.Close()
		b.conn = nil
	}
}

func (b *systemBus) property(ctx context.Context, dest, path, iface, name string) (dbus.Variant, error) {
	conn, err := b.get()
	if err != nil {
		return dbus.Variant{}, err
	}
	ctx, cancel := context.WithTimeout(ctx, dbusCallWait)
	defer cancel()
	var v dbus.Variant
	if err := conn.Object(dest, dbus.ObjectPath(path)).CallWithContext(ctx, propsGet, 0, iface, name).Store(&v); err != nil {
		return dbus.Variant{}, err
	}
	return v, nil
}

// systemPower asks UPower and falls back to /sys/class/power_supply.
type systemPower struct {
	bus   systemBus
	sysfs sysfsPower
}

func newSystemPower() *systemPower {
	return &systemPower{sysfs: sysfsPower{root: "/sys/class/power_supply"}}
}

func (p *systemPower) OnBattery(ctx context.Context) (bool, error) {
	v, err := p.bus.property(ctx, upowerDest, upowerPath, "org.freedesktop.UPower", "OnBattery")
	if err == nil {
		if b, ok := v.Value().(bool); ok {
			return b, nil
		}
	}
	return p.sysfs.OnBattery(ctx)
}

func (p *systemPower) close() { p.bus.close() }

// sysfsPower reads the kernel's power supply class: the machine is on battery
// when a battery is discharging and no mains or USB power supply is online.
type sysfsPower struct{ root string }

func (s sysfsPower) OnBattery(context.Context) (bool, error) {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return false, nil // no power supply information: a desktop
		}
		return false, err
	}
	discharging, external := false, false
	for _, e := range entries {
		dir := filepath.Join(s.root, e.Name())
		kind := readSysfs(dir, "type")
		switch kind {
		case "Battery":
			if readSysfs(dir, "scope") == "Device" { // a peripheral's battery, not the system's
				continue
			}
			discharging = discharging || readSysfs(dir, "status") == "Discharging"
		case "Mains", "USB", "USB_C", "USB_PD", "USB_PD_DRP", "USB_DCP", "USB_CDP", "USB_ACA":
			external = external || readSysfs(dir, "online") == "1"
		}
	}
	return discharging && !external, nil
}

func readSysfs(dir, name string) string {
	f, err := os.Open(filepath.Join(dir, name))
	if err != nil {
		return ""
	}
	defer f.Close()
	line, _ := bufio.NewReader(f).ReadString('\n')
	return strings.TrimSpace(line)
}

// logindActivity reads the login manager's idle hint of the user's sessions.
// Desktop environments set the hint when the screen saver or idle timeout
// starts (GNOME and KDE do this by themselves; a Wayland compositor needs an
// idle daemon that calls SetIdleHint). If no session has ever changed the
// hint within grace of the login, the desktop does not maintain it and the
// source reports "unknown" instead of pausing forever.
type logindActivity struct {
	bus   systemBus
	grace time.Duration
	now   func() time.Time
}

func newLogindActivity(grace time.Duration) *logindActivity {
	return &logindActivity{grace: grace, now: time.Now}
}

func (l *logindActivity) UserActive(ctx context.Context) (active, known bool, err error) {
	get := func(name string) (dbus.Variant, error) {
		return l.bus.property(ctx, logindDest, logindUser, "org.freedesktop.login1.User", name)
	}
	hint, err := get("IdleHint")
	if err != nil {
		return false, false, err
	}
	idle, ok := hint.Value().(bool)
	if !ok {
		return false, false, errors.New("login1 IdleHint is not a boolean")
	}
	if idle {
		return false, true, nil
	}
	since, err := get("IdleSinceHint")
	if err != nil {
		return false, false, err
	}
	if t, _ := since.Value().(uint64); t != 0 {
		return true, true, nil // the hint has been maintained at least once
	}
	login, err := get("Timestamp")
	if err != nil {
		return false, false, err
	}
	if t, _ := login.Value().(uint64); t != 0 && l.now().Sub(time.UnixMicro(int64(t))) < l.grace {
		return true, true, nil // too early to say that the desktop never sets the hint
	}
	return true, false, nil
}

func (l *logindActivity) close() { l.bus.close() }
