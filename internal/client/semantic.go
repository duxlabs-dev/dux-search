package client

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"time"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service"
)

// TopicModelProgress mirrors server.TopicModelProgress (the client does not
// import the server).
const TopicModelProgress = "model.progress"

func decode[T any](raw json.RawMessage, err error) (*T, error) {
	if err != nil {
		return nil, err
	}
	var out T
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// CollectionsSet replaces the content of a collection.
func CollectionsSet(name string, items []semantic.Item) (*service.SetResult, error) {
	if items == nil {
		items = []semantic.Item{}
	}
	return decode[service.SetResult](sendRequest("collections.set", map[string]any{"collection": name, "items": items}))
}

// CollectionsDelete removes a collection.
func CollectionsDelete(name string) (*service.DeleteResult, error) {
	return decode[service.DeleteResult](sendRequest("collections.delete", map[string]any{"collection": name}))
}

// SemanticStatus reads semantic.status.
func SemanticStatus() (*service.Status, error) {
	return decode[service.Status](sendRequest("semantic.status", nil))
}

// SemanticWarm loads the given towers (default text) into memory.
func SemanticWarm(modalities []string) (*service.Status, error) {
	params := map[string]any{}
	if len(modalities) > 0 {
		params["modalities"] = modalities
	}
	raw, err := sendRequest("semantic.warm", params)
	if err != nil {
		return nil, err
	}
	var resp struct {
		Embedder semantic.EmbedderStatus `json:"embedder"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		return nil, err
	}
	return &service.Status{Enabled: true, Embedder: resp.Embedder}, nil
}

// ModelStatusResult is the answer of model.status. Model is whatever the model
// manager reports about the installed files.
type ModelStatusResult struct {
	Model    json.RawMessage       `json:"model"`
	Download service.DownloadState `json:"download"`
}

// ModelStatus reads model.status.
func ModelStatus() (*ModelStatusResult, error) {
	return decode[ModelStatusResult](sendRequest("model.status", nil))
}

// ModelRemove deletes model towers.
func ModelRemove(towers []string) error {
	_, err := sendRequest("model.remove", map[string]any{"towers": towers})
	return err
}

// Stream is one open connection that can call methods and receive events.
type Stream struct {
	conn net.Conn
	r    *bufio.Reader
	next int
}

// OpenStream connects to the running service.
func OpenStream() (*Stream, error) {
	path, err := findRunningSocket()
	if err != nil {
		return nil, err
	}
	conn, err := net.Dial("unix", path)
	if err != nil {
		return nil, ErrServiceNotRunning
	}
	r := bufio.NewReader(conn)
	if _, err := r.ReadBytes('\n'); err != nil { // capabilities line
		conn.Close()
		return nil, fmt.Errorf("read capabilities: %w", err)
	}
	return &Stream{conn: conn, r: r}, nil
}

// Close closes the connection.
func (s *Stream) Close() error { return s.conn.Close() }

type frame struct {
	ID     int             `json:"id"`
	Result json.RawMessage `json:"result"`
	Error  string          `json:"error"`
	Event  string          `json:"event"`
	Data   json.RawMessage `json:"data"`
}

func (s *Stream) send(method string, params map[string]any) (int, error) {
	s.next++
	data, err := json.Marshal(map[string]any{"id": s.next, "method": method, "params": params})
	if err != nil {
		return 0, err
	}
	_, err = s.conn.Write(append(data, '\n'))
	return s.next, err
}

func (s *Stream) read() (frame, error) {
	line, err := s.r.ReadBytes('\n')
	if err != nil {
		return frame{}, err
	}
	var f frame
	return f, json.Unmarshal(line, &f)
}

// Call sends a request and waits for its response; events that arrive meanwhile
// go to onEvent (may be nil).
func (s *Stream) Call(method string, params map[string]any, onEvent func(topic string, data json.RawMessage)) (json.RawMessage, error) {
	id, err := s.send(method, params)
	if err != nil {
		return nil, err
	}
	for {
		f, err := s.read()
		if err != nil {
			return nil, err
		}
		switch {
		case f.Event != "":
			if onEvent != nil {
				onEvent(f.Event, f.Data)
			}
		case f.ID == id:
			if f.Error != "" {
				return nil, errors.New(f.Error)
			}
			return f.Result, nil
		}
	}
}

// ModelDownload starts a download and reports progress until it ends. It
// returns nil when the download finished and the failure otherwise. The
// download itself runs in the service, so closing this connection (or ^C in the
// CLI) does not stop it.
func ModelDownload(towers []string, onProgress func(service.ModelProgress)) error {
	s, err := OpenStream()
	if err != nil {
		return err
	}
	defer s.Close()
	if _, err := s.Call("subscribe", map[string]any{"topics": []string{TopicModelProgress}}, nil); err != nil {
		return err
	}
	// Events can overtake the response of model.download (a download that is
	// already complete ends at once), so they are handled while waiting for it.
	w := &progressWatch{onProgress: onProgress}
	if _, err := s.Call("model.download", map[string]any{"towers": towers}, w.event); err != nil {
		return err
	}
	if w.ended {
		return w.err
	}
	return s.WatchModelProgress(onProgress)
}

type progressWatch struct {
	onProgress func(service.ModelProgress)
	ended      bool
	err        error
}

func (w *progressWatch) event(topic string, data json.RawMessage) {
	if topic != TopicModelProgress || w.ended {
		return
	}
	var p service.ModelProgress
	if json.Unmarshal(data, &p) != nil {
		return
	}
	if w.onProgress != nil {
		w.onProgress(p)
	}
	switch p.State {
	case service.StateFinished:
		w.ended = true
	case service.StateFailed:
		w.ended, w.err = true, errors.New(p.Error)
	}
}

// WatchModelProgress reads model.progress events from a subscribed stream until
// the download finishes (nil) or fails (its error).
func (s *Stream) WatchModelProgress(onProgress func(service.ModelProgress)) error {
	w := &progressWatch{onProgress: onProgress}
	for !w.ended {
		_ = s.conn.SetReadDeadline(time.Now().Add(10 * time.Minute))
		f, err := s.read()
		if err != nil {
			return fmt.Errorf("lost the connection to dsearch while downloading: %w", err)
		}
		if f.Event != "" {
			w.event(f.Event, f.Data)
		}
	}
	return w.err
}
