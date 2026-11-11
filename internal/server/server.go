package server

import (
	"github.com/AvengeMedia/dankgo/ipc"
)

const (
	APIVersion  = 1
	maxLineSize = 10 * 1024 * 1024
)

func NewUnix(router *Router) *ipc.Server {
	return ipc.NewServer(ipc.Config{
		AppName:      "dsearch",
		APIVersion:   APIVersion,
		Capabilities: Capabilities,
		MaxLineSize:  maxLineSize,
		Bus:          router.Bus(),
	}, router.Handle)
}
