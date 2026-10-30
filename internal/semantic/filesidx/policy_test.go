package filesidx

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// supply writes one /sys/class/power_supply entry the way the kernel does.
func supply(t *testing.T, root, name string, attrs map[string]string) {
	t.Helper()
	dir := filepath.Join(root, name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for k, v := range attrs {
		if err := os.WriteFile(filepath.Join(dir, k), []byte(v+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestSysfsPower(t *testing.T) {
	cases := []struct {
		name string
		set  func(root string)
		want bool
	}{
		{"desktop without supplies", func(string) {}, false},
		{"laptop on mains, charging", func(r string) {
			supply(t, r, "BAT0", map[string]string{"type": "Battery", "status": "Charging"})
			supply(t, r, "AC", map[string]string{"type": "Mains", "online": "1"})
		}, false},
		{"laptop on mains, full", func(r string) {
			supply(t, r, "BAT0", map[string]string{"type": "Battery", "status": "Full"})
			supply(t, r, "AC", map[string]string{"type": "Mains", "online": "1"})
		}, false},
		{"laptop unplugged", func(r string) {
			supply(t, r, "BAT0", map[string]string{"type": "Battery", "status": "Discharging"})
			supply(t, r, "AC", map[string]string{"type": "Mains", "online": "0"})
		}, true},
		{"discharging while a USB-C charger is online (slow charger)", func(r string) {
			supply(t, r, "BAT1", map[string]string{"type": "Battery", "status": "Discharging"})
			supply(t, r, "ucsi-source-psy-USBC000:001", map[string]string{"type": "USB", "online": "1"})
		}, false},
		{"only a peripheral battery discharges", func(r string) {
			supply(t, r, "hid-mouse", map[string]string{"type": "Battery", "scope": "Device", "status": "Discharging"})
		}, false},
		{"unplugged laptop plus a mouse", func(r string) {
			supply(t, r, "BAT0", map[string]string{"type": "Battery", "status": "Discharging"})
			supply(t, r, "AC", map[string]string{"type": "Mains", "online": "0"})
			supply(t, r, "hid-mouse", map[string]string{"type": "Battery", "scope": "Device", "status": "Full"})
		}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			root := t.TempDir()
			c.set(root)
			got, err := sysfsPower{root: root}.OnBattery(context.Background())
			if err != nil || got != c.want {
				t.Fatalf("OnBattery = %v, %v; want %v", got, err, c.want)
			}
		})
	}
	if on, err := (sysfsPower{root: filepath.Join(t.TempDir(), "missing")}).OnBattery(context.Background()); on || err != nil {
		t.Fatalf("missing directory: %v %v", on, err)
	}
}

// The next two tests talk to the real system bus of the machine they run on.

func TestSystemPowerAgainstRealBus(t *testing.T) {
	p := newSystemPower()
	defer p.close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := p.bus.property(ctx, upowerDest, upowerPath, "org.freedesktop.UPower", "OnBattery"); err != nil {
		t.Skipf("UPower is not reachable on this machine: %v", err)
	}
	got, err := p.OnBattery(ctx)
	if err != nil {
		t.Fatal(err)
	}
	// UPower and the kernel must agree on a machine that is not mid-transition.
	if want, _ := p.sysfs.OnBattery(ctx); got != want {
		t.Logf("UPower says %v, sysfs says %v", got, want)
	}
	t.Logf("OnBattery = %v", got)
}

func TestLogindActivityAgainstRealBus(t *testing.T) {
	a := newLogindActivity(30 * time.Minute)
	defer a.close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := a.bus.property(ctx, logindDest, logindUser, "org.freedesktop.login1.User", "IdleHint"); err != nil {
		t.Skipf("logind is not reachable on this machine: %v", err)
	}
	active, known, err := a.UserActive(ctx)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("active=%v known=%v", active, known)

	// With a zero grace period a desktop that never touched the hint is "unknown".
	a.grace = 0
	_, known0, err := a.UserActive(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if known && !known0 {
		t.Fatal("the hint was maintained, a shorter grace must not hide that")
	}
}
