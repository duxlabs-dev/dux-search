package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/AvengeMedia/dankgo/ipc"
	"github.com/AvengeMedia/dankgo/ipc/params"
	"github.com/duxlabs-dev/dux-search/internal/indexer"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service"
)

// TopicModelProgress is the subscription topic of model download progress:
// {"tower":"text","bytes":123,"total":456,"state":"downloading"}. A download
// ends with {"state":"finished"} or {"state":"failed","error":"..."}.
const TopicModelProgress = "model.progress"

// Capabilities are announced by this build (version --json and the socket
// handshake). They say what the binary can do, not whether it is enabled;
// semantic.status says that.
var Capabilities = []string{"semantic", "collections"}

func (r *Router) handleSemanticSearch(ctx context.Context, w *ipc.ConnWriter, req ipc.Request, opts *indexer.SearchOptions) {
	hr := service.HybridRequest{
		Collections:     params.StringSlice(req.Params, "collections"),
		CollectionLimit: params.IntOpt(req.Params, "collection_limit", 0),
	}
	res, err := r.semantic.HybridSearch(ctx, r.indexer, opts, hr)
	if err != nil {
		ipc.RespondError(w, req.ID, fmt.Sprintf("search failed: %v", err))
		return
	}
	ipc.Respond(w, req.ID, res)
}

func (r *Router) handleCollectionsSet(ctx context.Context, w *ipc.ConnWriter, req ipc.Request) {
	name, err := params.StringNonEmpty(req.Params, "collection")
	if err != nil {
		ipc.RespondError(w, req.ID, "collection parameter required")
		return
	}
	raw, ok := params.Any(req.Params, "items")
	if !ok {
		ipc.RespondError(w, req.ID, "items parameter required")
		return
	}
	items, err := decodeItems(raw)
	if err != nil {
		ipc.RespondError(w, req.ID, err.Error())
		return
	}
	res, err := r.semantic.SetCollection(ctx, name, items)
	if err != nil {
		ipc.RespondError(w, req.ID, fmt.Sprintf("collections.set: %v", err))
		return
	}
	ipc.Respond(w, req.ID, res)
}

// decodeItems converts the loosely typed request value into items; unknown item
// fields are ignored so clients can attach their own metadata.
func decodeItems(raw any) ([]semantic.Item, error) {
	if raw == nil {
		return nil, errors.New("items must be an array")
	}
	if _, isList := raw.([]any); !isList {
		return nil, errors.New("items must be an array")
	}
	buf, err := json.Marshal(raw)
	if err != nil {
		return nil, fmt.Errorf("items: %w", err)
	}
	var items []semantic.Item
	if err := json.Unmarshal(buf, &items); err != nil {
		return nil, fmt.Errorf("items: %w", err)
	}
	return items, nil
}

func (r *Router) handleCollectionsDelete(w *ipc.ConnWriter, req ipc.Request) {
	name, err := params.StringNonEmpty(req.Params, "collection")
	if err != nil {
		ipc.RespondError(w, req.ID, "collection parameter required")
		return
	}
	res, err := r.semantic.DeleteCollection(name)
	if err != nil {
		ipc.RespondError(w, req.ID, fmt.Sprintf("collections.delete: %v", err))
		return
	}
	ipc.Respond(w, req.ID, res)
}

func (r *Router) handleSemanticStatus(w *ipc.ConnWriter, req ipc.Request) {
	ipc.Respond(w, req.ID, r.semantic.Status())
}

// warmTimeout bounds a warm-up: loading the model from disk is the slow part.
const warmTimeout = 60 * time.Second

func (r *Router) handleSemanticWarm(ctx context.Context, w *ipc.ConnWriter, req ipc.Request) {
	mods, err := service.ParseModalities(params.StringSlice(req.Params, "modalities"))
	if err != nil {
		ipc.RespondError(w, req.ID, err.Error())
		return
	}
	wctx, cancel := context.WithTimeout(ctx, warmTimeout)
	defer cancel()
	if err := r.semantic.Warm(wctx, mods...); err != nil {
		ipc.RespondError(w, req.ID, fmt.Sprintf("semantic.warm: %v", err))
		return
	}
	ipc.Respond(w, req.ID, map[string]any{"status": "warm", "embedder": r.semantic.Status().Embedder})
}

func (r *Router) handleModelStatus(w *ipc.ConnWriter, req ipc.Request) {
	st, err := r.semantic.ModelStatus()
	if err != nil {
		ipc.RespondError(w, req.ID, fmt.Sprintf("model.status: %v", err))
		return
	}
	ipc.Respond(w, req.ID, map[string]any{"model": st, "download": r.semantic.Status().Download})
}

func (r *Router) towers(w *ipc.ConnWriter, req ipc.Request) ([]semantic.Modality, bool) {
	towers, err := service.ParseModalities(params.StringSlice(req.Params, "towers"))
	if err == nil && len(towers) == 0 {
		err = errors.New("towers parameter required (text, image, audio)")
	}
	if err != nil {
		ipc.RespondError(w, req.ID, err.Error())
		return nil, false
	}
	return towers, true
}

func (r *Router) handleModelDownload(w *ipc.ConnWriter, req ipc.Request) {
	towers, ok := r.towers(w, req)
	if !ok {
		return
	}
	if err := r.semantic.ModelDownload(towers); err != nil {
		ipc.RespondError(w, req.ID, fmt.Sprintf("model.download: %v", err))
		return
	}
	ipc.Respond(w, req.ID, map[string]any{"status": "download started", "towers": towers})
}

func (r *Router) handleModelRemove(w *ipc.ConnWriter, req ipc.Request) {
	towers, ok := r.towers(w, req)
	if !ok {
		return
	}
	if err := r.semantic.ModelRemove(towers); err != nil {
		ipc.RespondError(w, req.ID, fmt.Sprintf("model.remove: %v", err))
		return
	}
	ipc.Respond(w, req.ID, map[string]any{"status": "removed", "towers": towers})
}
