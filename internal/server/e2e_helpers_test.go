package server

import (
	"encoding/json"
	"net"
	"net/url"
	"testing"

	"github.com/AvengeMedia/dankgo/ipc"
	"github.com/duxlabs-dev/dux-search/internal/indexer"
	"github.com/stretchr/testify/require"
)

type searchOpts = indexer.SearchOptions

type net_ struct{}

func (*net_) dial(path string) (net.Conn, error) { return net.Dial("unix", path) }

func urlq(s string) string { return url.QueryEscape(s) }

// rawCall sends one request over the socket and returns the raw result.
func rawCall(method string, params map[string]any) (json.RawMessage, error) {
	path, err := findSocket()
	if err != nil {
		return nil, err
	}
	c, err := ipc.Dial(path)
	if err != nil {
		return nil, err
	}
	defer c.Close()
	resp, err := c.Call(ipc.Request{ID: 1, Method: method, Params: params})
	if err != nil {
		return nil, err
	}
	if resp.Error != "" {
		return nil, &callError{resp.Error}
	}
	return json.Marshal(resp.Result)
}

type callError struct{ msg string }

func (e *callError) Error() string { return e.msg }

func findSocket() (string, error) {
	for _, name := range []string{"dsearch", "danksearch"} {
		if p, err := ipc.FindRunningSocket(name); err == nil {
			return p, nil
		}
	}
	return "", &callError{"no socket"}
}

func rawSearch(t *testing.T, params map[string]any) json.RawMessage {
	t.Helper()
	raw, err := rawCall("search", params)
	require.NoError(t, err)
	return raw
}
