package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseItems(t *testing.T) {
	want := []semantic.Item{{ID: "dark", Title: "Dark theme", Text: "Switch to dark", Keywords: []string{"night"}}}
	for name, in := range map[string]string{
		"array":   `[{"id":"dark","title":"Dark theme","text":"Switch to dark","keywords":["night"]}]`,
		"wrapped": `{"items":[{"id":"dark","title":"Dark theme","text":"Switch to dark","keywords":["night"]}]}`,
		"padded":  "\n  " + `[{"id":"dark","title":"Dark theme","text":"Switch to dark","keywords":["night"]}]` + "\n",
	} {
		got, err := parseItems([]byte(in))
		require.NoError(t, err, name)
		assert.Equal(t, want, got, name)
	}

	got, err := parseItems([]byte(`[]`))
	require.NoError(t, err)
	assert.Empty(t, got, "an empty set is valid: it clears the collection")

	for name, in := range map[string]string{
		"empty":            "",
		"whitespace":       " \n ",
		"not json":         "dark theme",
		"object no items":  `{"foo":1}`,
		"item wrong shape": `[{"id":5}]`,
		"truncated":        `[{"id":"a"`,
	} {
		_, err := parseItems([]byte(in))
		assert.Error(t, err, name)
	}
}

func TestTowerNames(t *testing.T) {
	got, err := towerNames(nil, false)
	require.NoError(t, err)
	assert.Equal(t, []string{"text"}, got, "text is the default tower")

	got, err = towerNames(nil, true)
	require.NoError(t, err)
	assert.Equal(t, []string{"text", "image", "audio"}, got)

	got, err = towerNames([]string{"audio", "text", "audio"}, false)
	require.NoError(t, err)
	assert.Equal(t, []string{"audio", "text"}, got, "deduplicated, in a fixed order")

	_, err = towerNames([]string{"video"}, false)
	assert.Error(t, err)
}

func TestDescribeProgress(t *testing.T) {
	assert.Equal(t, "downloading  1.0 KiB / 4.0 KiB (25%)", describeProgress(service.ModelProgress{State: "downloading", Bytes: 1024, Total: 4096}))
	assert.Equal(t, "verifying", describeProgress(service.ModelProgress{State: "verifying"}))
}

func TestProgressPrinterOnAPipe(t *testing.T) {
	var buf bytes.Buffer
	p := &progressPrinter{w: &buf, last: map[semantic.Modality]string{}}
	for _, ev := range []service.ModelProgress{
		{Tower: "text", State: "downloading", Bytes: 1, Total: 10},
		{Tower: "text", State: "downloading", Bytes: 5, Total: 10},
		{Tower: "text", State: "verified", Bytes: 10, Total: 10},
		{State: "finished"},
	} {
		p.update(ev)
	}
	p.finish()
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	require.Len(t, lines, 2, "a pipe gets one line per state change, not one per chunk")
	assert.Contains(t, lines[0], "downloading")
	assert.Contains(t, lines[1], "verified")
	assert.NotContains(t, buf.String(), "\r")
}

func TestProgressPrinterOnATerminal(t *testing.T) {
	var buf bytes.Buffer
	p := &progressPrinter{w: &buf, tty: true, last: map[semantic.Modality]string{}}
	p.update(service.ModelProgress{Tower: "image", State: "downloading", Bytes: 1, Total: 10})
	p.update(service.ModelProgress{Tower: "image", State: "downloading", Bytes: 9, Total: 10})
	p.finish()
	assert.Equal(t, 2, strings.Count(buf.String(), "\r\033[K"), "the line is redrawn in place")
	assert.True(t, strings.HasSuffix(buf.String(), "\n"), "and ends with a newline")
}

func TestJoinModalities(t *testing.T) {
	assert.Equal(t, "none", joinModalities(nil))
	assert.Equal(t, "text, image", joinModalities([]semantic.Modality{"text", "image"}))
}
