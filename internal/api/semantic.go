package api

import (
	"context"
	"errors"
	"time"

	"github.com/danielgtaylor/huma/v2"
	"github.com/duxlabs-dev/dux-search/internal/semantic"
	"github.com/duxlabs-dev/dux-search/internal/semantic/service"
)

type CollectionPath struct {
	Name string `path:"name" doc:"Collection name (letters, digits and . _ : -)" example:"shell.actions"`
}

type CollectionSetInput struct {
	CollectionPath
	Body struct {
		Items []semantic.Item `json:"items" doc:"The complete new content of the collection"`
	}
}

type CollectionSetOutput struct{ Body service.SetResult }
type CollectionDeleteOutput struct{ Body service.DeleteResult }
type SemanticStatusOutput struct{ Body service.Status }

type WarmInput struct {
	Modalities []string `query:"modalities" doc:"Towers to load (text, image, audio); default text" example:"text"`
}

type WarmOutput struct {
	Body struct {
		Status   string                  `json:"status" example:"warm"`
		Embedder semantic.EmbedderStatus `json:"embedder"`
	}
}

type ModelStatusOutput struct {
	Body struct {
		Model    any                   `json:"model" doc:"Installed model files as the model manager reports them"`
		Download service.DownloadState `json:"download"`
	}
}

type TowersInput struct {
	Body struct {
		Towers []string `json:"towers" minItems:"1" doc:"text, image and/or audio" example:"[\"text\"]"`
	}
}

type TowersOutput struct {
	Body struct {
		Status string   `json:"status" example:"download started"`
		Towers []string `json:"towers"`
	}
}

// semanticError maps service errors to HTTP statuses.
func semanticError(op string, err error) error {
	switch {
	case errors.Is(err, service.ErrDisabled), errors.Is(err, service.ErrDownloadRunning):
		return huma.Error409Conflict(op+": "+err.Error(), err)
	case errors.Is(err, service.ErrNoModelManager):
		return huma.Error501NotImplemented(op+": "+err.Error(), err)
	case errors.Is(err, semantic.ErrModelUnavailable):
		return huma.Error503ServiceUnavailable(op+": "+err.Error(), err)
	default:
		return huma.Error400BadRequest(op+": "+err.Error(), err)
	}
}

func towerList(in []string) ([]semantic.Modality, error) {
	towers, err := service.ParseModalities(in)
	if err == nil && len(towers) == 0 {
		err = errors.New("no towers given")
	}
	return towers, err
}

func registerSemanticHandlers(srv *Server, api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "semanticStatus",
		Summary:     "Semantic search status",
		Description: "Embedder, model, collections, background indexing and model download state",
		Method:      "GET",
		Path:        "/semantic/status",
		Tags:        []string{"Semantic"},
	}, func(ctx context.Context, _ *struct{}) (*SemanticStatusOutput, error) {
		return &SemanticStatusOutput{Body: srv.semantic().Status()}, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "semanticWarm",
		Summary:     "Load the model ahead of use",
		Method:      "POST",
		Path:        "/semantic/warm",
		Tags:        []string{"Semantic"},
	}, func(ctx context.Context, in *WarmInput) (*WarmOutput, error) {
		mods, err := service.ParseModalities(in.Modalities)
		if err != nil {
			return nil, semanticError("warm", err)
		}
		wctx, cancel := context.WithTimeout(ctx, 25*time.Second)
		defer cancel()
		if err := srv.semantic().Warm(wctx, mods...); err != nil {
			return nil, semanticError("warm", err)
		}
		out := &WarmOutput{}
		out.Body.Status = "warm"
		out.Body.Embedder = srv.semantic().Status().Embedder
		return out, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "collectionSet",
		Summary:     "Replace a collection",
		Description: "Replaces the whole content of a collection. Items with an unchanged content hash are not embedded again; with no model the items are stored and embedded when it is installed.",
		Method:      "PUT",
		Path:        "/collections/{name}",
		Tags:        []string{"Semantic"},
	}, func(ctx context.Context, in *CollectionSetInput) (*CollectionSetOutput, error) {
		res, err := srv.semantic().SetCollection(ctx, in.Name, in.Body.Items)
		if err != nil {
			return nil, semanticError("collections.set", err)
		}
		return &CollectionSetOutput{Body: res}, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "collectionDelete",
		Summary:     "Delete a collection",
		Method:      "DELETE",
		Path:        "/collections/{name}",
		Tags:        []string{"Semantic"},
	}, func(ctx context.Context, in *CollectionPath) (*CollectionDeleteOutput, error) {
		res, err := srv.semantic().DeleteCollection(in.Name)
		if err != nil {
			return nil, semanticError("collections.delete", err)
		}
		return &CollectionDeleteOutput{Body: res}, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "modelStatus",
		Summary:     "Model installation status",
		Method:      "GET",
		Path:        "/model/status",
		Tags:        []string{"Semantic"},
	}, func(ctx context.Context, _ *struct{}) (*ModelStatusOutput, error) {
		st, err := srv.semantic().ModelStatus()
		if err != nil {
			return nil, semanticError("model.status", err)
		}
		out := &ModelStatusOutput{}
		out.Body.Model = st
		out.Body.Download = srv.semantic().Status().Download
		return out, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "modelDownload",
		Summary:     "Download model towers",
		Description: "Starts the download in the background; poll /semantic/status (download) for progress.",
		Method:      "POST",
		Path:        "/model/download",
		Tags:        []string{"Semantic"},
	}, func(ctx context.Context, in *TowersInput) (*TowersOutput, error) {
		towers, err := towerList(in.Body.Towers)
		if err != nil {
			return nil, semanticError("model.download", err)
		}
		if err := srv.semantic().ModelDownload(towers); err != nil {
			return nil, semanticError("model.download", err)
		}
		out := &TowersOutput{}
		out.Body.Status, out.Body.Towers = "download started", in.Body.Towers
		return out, nil
	})

	huma.Register(api, huma.Operation{
		OperationID: "modelRemove",
		Summary:     "Remove model towers",
		Method:      "POST",
		Path:        "/model/remove",
		Tags:        []string{"Semantic"},
	}, func(ctx context.Context, in *TowersInput) (*TowersOutput, error) {
		towers, err := towerList(in.Body.Towers)
		if err != nil {
			return nil, semanticError("model.remove", err)
		}
		if err := srv.semantic().ModelRemove(towers); err != nil {
			return nil, semanticError("model.remove", err)
		}
		out := &TowersOutput{}
		out.Body.Status, out.Body.Towers = "removed", in.Body.Towers
		return out, nil
	})
}
