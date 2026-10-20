package server

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/AvengeMedia/dankgo/ipc"
	bleve "github.com/blevesearch/bleve/v2"
	"github.com/duxlabs-dev/dux-search/internal/config"
	"github.com/duxlabs-dev/dux-search/internal/indexer"
	mocks_net "github.com/duxlabs-dev/dux-search/internal/mocks/net"
	mocks_server "github.com/duxlabs-dev/dux-search/internal/mocks/server"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/suite"
)

type RouterSuite struct {
	suite.Suite
	indexer *mocks_server.MockIndexerInterface
	watcher *mocks_server.MockWatcherInterface
	router  *Router
}

func TestRouterSuite(t *testing.T) {
	suite.Run(t, new(RouterSuite))
}

func (s *RouterSuite) SetupTest() {
	s.indexer = mocks_server.NewMockIndexerInterface(s.T())
	s.watcher = mocks_server.NewMockWatcherInterface(s.T())
	s.router = NewRouter(s.indexer, s.watcher, &config.Config{IndexPaths: []config.IndexPath{
		{Path: "/home/u", MaxDepth: 6, ExcludeHidden: true, ExcludeDirs: []string{"node_modules"}},
		{Path: "/srv/data"},
	}})
}

func (s *RouterSuite) route(req ipc.Request) []byte {
	buf := &bytes.Buffer{}
	conn := mocks_net.NewMockConn(s.T())
	conn.EXPECT().SetWriteDeadline(mock.Anything).Return(nil).Maybe()
	conn.EXPECT().Write(mock.Anything).RunAndReturn(buf.Write).Maybe()
	s.router.Handle(context.Background(), ipc.NewConnWriter(conn), req, nil)
	return buf.Bytes()
}

func (s *RouterSuite) TestSearch() {
	s.indexer.EXPECT().SearchWithOptions(mock.Anything).Return(&bleve.SearchResult{Total: 5}, nil).Once()

	out := s.route(ipc.Request{
		ID:     2,
		Method: "search",
		Params: map[string]any{"query": "test", "limit": float64(10)},
	})

	var resp ipc.Response[bleve.SearchResult]
	s.Require().NoError(json.Unmarshal(out, &resp))
	s.Equal(2, resp.ID)
	s.Require().NotNil(resp.Result)
	s.Equal(uint64(5), resp.Result.Total)
}

func (s *RouterSuite) TestStats() {
	s.indexer.EXPECT().Stats().Return(&config.IndexStats{TotalFiles: 100}).Once()

	out := s.route(ipc.Request{ID: 3, Method: "stats"})

	var resp ipc.Response[config.IndexStats]
	s.Require().NoError(json.Unmarshal(out, &resp))
	s.Require().NotNil(resp.Result)
	s.EqualValues(100, resp.Result.TotalFiles)
}

func (s *RouterSuite) TestInfo() {
	s.indexer.EXPECT().Stats().Return(&config.IndexStats{SchemaVersion: 4, Phase: indexer.PhaseSyncing}).Once()

	out := s.route(ipc.Request{ID: 10, Method: "info"})

	var resp ipc.Response[Info]
	s.Require().NoError(json.Unmarshal(out, &resp))
	s.Require().NotNil(resp.Result)
	s.Equal(4, resp.Result.SchemaVersion)
	s.Equal(indexer.SchemaVersion, resp.Result.ExpectedSchemaVersion)
	s.Equal(indexer.PhaseSyncing, resp.Result.Phase)
	s.Require().Len(resp.Result.IndexPaths, 2)
	s.Equal(IndexPathInfo{Path: "/home/u", MaxDepth: 6, ExcludeHidden: true, ExcludeDirs: []string{"node_modules"}}, resp.Result.IndexPaths[0])
	s.Equal(IndexPathInfo{Path: "/srv/data", ExcludeDirs: []string{}}, resp.Result.IndexPaths[1])
}

func (s *RouterSuite) TestInfoWithoutStats() {
	s.indexer.EXPECT().Stats().Return(nil).Once()

	out := s.route(ipc.Request{ID: 11, Method: "info"})

	var resp ipc.Response[Info]
	s.Require().NoError(json.Unmarshal(out, &resp))
	s.Equal(indexer.PhaseIdle, resp.Result.Phase)
	s.Equal(0, resp.Result.SchemaVersion)
}

func (s *RouterSuite) TestWatchStart() {
	s.watcher.EXPECT().IsRunning().Return(false).Once()
	s.watcher.EXPECT().Start().Return(nil).Once()

	out := s.route(ipc.Request{ID: 4, Method: "watch.start"})

	var resp ipc.Response[map[string]string]
	s.Require().NoError(json.Unmarshal(out, &resp))
	s.Require().NotNil(resp.Result)
	s.Equal("watcher started", (*resp.Result)["status"])
}

func (s *RouterSuite) TestWatchStop() {
	s.watcher.EXPECT().IsRunning().Return(true).Once()
	s.watcher.EXPECT().Stop().Return(nil).Once()

	out := s.route(ipc.Request{ID: 5, Method: "watch.stop"})

	var resp ipc.Response[map[string]string]
	s.Require().NoError(json.Unmarshal(out, &resp))
	s.Require().NotNil(resp.Result)
	s.Equal("watcher stopped", (*resp.Result)["status"])
}

func (s *RouterSuite) TestWatchStatus() {
	tests := []struct {
		name     string
		running  bool
		expected string
	}{
		{"running", true, "running"},
		{"stopped", false, "stopped"},
	}

	for _, tt := range tests {
		s.Run(tt.name, func() {
			s.watcher.EXPECT().IsRunning().Return(tt.running).Once()

			out := s.route(ipc.Request{ID: 6, Method: "watch.status"})

			var resp ipc.Response[map[string]string]
			s.Require().NoError(json.Unmarshal(out, &resp))
			s.Require().NotNil(resp.Result)
			s.Equal(tt.expected, (*resp.Result)["status"])
		})
	}
}

func (s *RouterSuite) TestReindex() {
	s.indexer.EXPECT().Stats().Return(&config.IndexStats{Phase: indexer.PhaseIdle}).Once()
	s.indexer.EXPECT().ReindexAll().Return(nil).Maybe()

	out := s.route(ipc.Request{ID: 7, Method: "reindex"})
	time.Sleep(50 * time.Millisecond)

	var resp ipc.Response[map[string]string]
	s.Require().NoError(json.Unmarshal(out, &resp))
	s.Require().NotNil(resp.Result)
	s.Equal("reindexing started", (*resp.Result)["status"])
}

func (s *RouterSuite) TestReindexRejectedWhileBusy() {
	s.indexer.EXPECT().Stats().Return(&config.IndexStats{Phase: indexer.PhaseSyncing}).Twice()

	for _, method := range []string{"reindex", "sync"} {
		out := s.route(ipc.Request{ID: 9, Method: method})

		var resp ipc.Response[any]
		s.Require().NoError(json.Unmarshal(out, &resp))
		s.Contains(resp.Error, "already in progress")
	}
}

func (s *RouterSuite) TestUnknownMethod() {
	out := s.route(ipc.Request{ID: 8, Method: "unknown"})

	var resp ipc.Response[any]
	s.Require().NoError(json.Unmarshal(out, &resp))
	s.NotEmpty(resp.Error)
}
