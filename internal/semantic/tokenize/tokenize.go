// Package tokenize is a pure Go tokenizer for the EmbeddingGemma 2 model. It
// reproduces the Hugging Face tokenizer.json pipeline of the model (replace
// spaces by U+2581, BPE with byte fallback, <bos> ... <eos> template, special
// tokens matched inside the text) on top of the SentencePiece model file.
package tokenize

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode/utf8"

	sentencepiece "github.com/eliben/go-sentencepiece"
	"google.golang.org/protobuf/encoding/protowire"
)

// Token ids fixed by the model (config.json text_config and tokenizer.json).
const (
	PadID = 0
	EOSID = 1
	BOSID = 2
)

// special is one entry of the "added_tokens" list in tokenizer.json. The
// tokenizer matches these literally anywhere in the input and never splits or
// normalizes them (all are special, normalized=false, no strip flags).
type special struct {
	text string
	id   int32
}

// specials is added_tokens of the model's tokenizer.json. TestSpecialsMatchTokenizerJSON
// keeps it in sync with the file shipped in the model directory.
var specials = []special{
	{"<pad>", 0}, {"<eos>", 1}, {"<bos>", 2}, {"<unk>", 3}, {"<mask>", 4},
	{"<|tool>", 46}, {"<tool|>", 47}, {"<|tool_call>", 48}, {"<tool_call|>", 49},
	{"<|tool_response>", 50}, {"<tool_response|>", 51}, {`<|"|>`, 52}, {"<|think|>", 98},
	{"<|channel>", 100}, {"<channel|>", 101}, {"<|turn>", 105}, {"<turn|>", 106},
	{"<|image>", 255999}, {"<|audio>", 256000}, {"<|image|>", 258880}, {"<|audio|>", 258881},
	{"<image|>", 258882}, {"<audio|>", 258883}, {"<|video|>", 258884},
}

// Tokenizer encodes text into model token ids. It is safe for concurrent use.
type Tokenizer struct {
	sp *sentencepiece.Processor
}

// Load reads tokenizer.model from path.
func Load(path string) (*Tokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return New(f)
}

// New reads a SentencePiece model from r.
func New(r io.Reader) (*Tokenizer, error) {
	raw, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("tokenizer model: %w", err)
	}
	raw, err = demoteUserDefined(raw)
	if err != nil {
		return nil, fmt.Errorf("tokenizer model: %w", err)
	}
	sp, err := sentencepiece.NewProcessor(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("tokenizer model: %w", err)
	}
	return &Tokenizer{sp: sp}, nil
}

// Encode returns <bos> + tokens(text) + <eos>, like tokenizer.json's
// post_processor does with add_special_tokens=true.
func (t *Tokenizer) Encode(text string) []int32 {
	ids := make([]int32, 0, len(text)/3+3)
	ids = append(ids, BOSID)
	ids = t.appendRaw(ids, text)
	return append(ids, EOSID)
}

// EncodeRaw returns the tokens of text without <bos> and <eos>.
func (t *Tokenizer) EncodeRaw(text string) []int32 {
	return t.appendRaw(nil, text)
}

func (t *Tokenizer) appendRaw(ids []int32, text string) []int32 {
	if !utf8.ValidString(text) {
		text = strings.ToValidUTF8(text, "�")
	}
	for len(text) > 0 {
		i, sp := nextSpecial(text)
		if i < 0 {
			return t.appendPlain(ids, text)
		}
		ids = t.appendPlain(ids, text[:i])
		ids = append(ids, sp.id)
		text = text[i+len(sp.text):]
	}
	return ids
}

func (t *Tokenizer) appendPlain(ids []int32, text string) []int32 {
	if text == "" {
		return ids
	}
	for _, tok := range t.sp.Encode(text) {
		ids = append(ids, int32(tok.ID))
	}
	return ids
}

// nextSpecial finds the leftmost special token in text, preferring the
// longest at one position.
func nextSpecial(text string) (int, special) {
	for from := 0; ; {
		j := strings.IndexByte(text[from:], '<')
		if j < 0 {
			return -1, special{}
		}
		at := from + j
		var best special
		for _, s := range specials {
			if len(s.text) > len(best.text) && strings.HasPrefix(text[at:], s.text) {
				best = s
			}
		}
		if best.text != "" {
			return at, best
		}
		from = at + 1
	}
}

// Field numbers and enum values of sentencepiece_model.proto.
const (
	fieldPieces      = 1 // ModelProto.pieces
	fieldPieceType   = 3 // SentencePiece.type
	pieceTypeNormal  = 1
	pieceTypeUserDef = 4
)

// demoteUserDefined rewrites every USER_DEFINED piece of a serialized
// ModelProto as NORMAL. SentencePiece matches user-defined pieces ("<b>",
// "<unused5>", "[multimodal]", ...) as atomic symbols before merging, but the
// model's tokenizer.json has no such rule: there they are ordinary vocabulary
// entries reached through BPE merges, and tokenizer.json is the reference the
// embeddings were validated with. The pieces keep their ids and scores.
func demoteUserDefined(model []byte) ([]byte, error) {
	out := make([]byte, 0, len(model))
	for rest := model; len(rest) > 0; {
		num, typ, n := protowire.ConsumeTag(rest)
		if n < 0 {
			return nil, protowire.ParseError(n)
		}
		l := protowire.ConsumeFieldValue(num, typ, rest[n:])
		if l < 0 {
			return nil, protowire.ParseError(l)
		}
		field := rest[:n+l]
		rest = rest[n+l:]
		if num != fieldPieces || typ != protowire.BytesType {
			out = append(out, field...)
			continue
		}
		body, _ := protowire.ConsumeBytes(field[n:])
		nb, err := demotePiece(body)
		if err != nil {
			return nil, err
		}
		out = protowire.AppendTag(out, num, typ)
		out = protowire.AppendBytes(out, nb)
	}
	return out, nil
}

func demotePiece(piece []byte) ([]byte, error) {
	out := make([]byte, 0, len(piece))
	for rest := piece; len(rest) > 0; {
		num, typ, n := protowire.ConsumeTag(rest)
		if n < 0 {
			return nil, protowire.ParseError(n)
		}
		l := protowire.ConsumeFieldValue(num, typ, rest[n:])
		if l < 0 {
			return nil, protowire.ParseError(l)
		}
		if num == fieldPieceType && typ == protowire.VarintType {
			v, _ := protowire.ConsumeVarint(rest[n:])
			if v == pieceTypeUserDef {
				out = protowire.AppendTag(out, num, typ)
				out = protowire.AppendVarint(out, pieceTypeNormal)
				rest = rest[n+l:]
				continue
			}
		}
		out = append(out, rest[:n+l]...)
		rest = rest[n+l:]
	}
	return out, nil
}
