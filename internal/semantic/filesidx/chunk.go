package filesidx

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Retrieval quality is best when a chunk is a coherent passage of about 512
// tokens. The tokenizer lives in another process, so chunk sizes are budgeted
// with a per-character estimate of the SentencePiece token count (tokenCost).
// The weights come from a non-negative least squares fit against the real
// EmbeddingGemma 2 tokenizer (Go and Python sources, Markdown, English, Turkish,
// German, French, Russian, Chinese and emoji text). Per kind of text the real
// count is 0.93x to 1.12x of the estimate on average; single files range from
// about 0.55x to 1.7x. testdata/fit_token_cost.py prints that table again and
// TestChunkTokenBudget checks the real size of every chunk of the test corpus
// (mean ~400, max below 500 real tokens).
const (
	// targetTokens is the estimated size a chunk is packed to. With the
	// overlap the largest chunk is about 510 estimated tokens, i.e. up to
	// roughly 590 real tokens in the worst case, far below the model's 2048
	// token context and close to the 512 token sweet spot.
	targetTokens = 450
	// overlapTokens is carried from one chunk into the next so a sentence cut
	// at a boundary is still whole in one of the two chunks.
	overlapTokens = 60
)

// Estimate weights, thousandths of a token per rune.
const (
	costASCIIWord   = 190  // letters and digits
	costASCIIPunct  = 1200 // code punctuation is rarely merged
	costNewline     = 1300 // includes the indentation that usually follows
	costTab         = 500
	costSpace       = 0    // folded into the following word token
	costLatinExt    = 1800 // Latin-1/Extended letters such as Turkish ç ğ ı ö ş ü split into pieces
	costOtherLetter = 220  // Cyrillic, Greek, Arabic and other alphabets
	costCJK         = 640
	costOther       = 1000 // symbols, typographic quotes, emoji
)

// tokenCost estimates the token count of one rune in thousandths of a token.
func tokenCost(r rune) int {
	switch {
	case r < utf8.RuneSelf:
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			return costASCIIWord
		case r == ' ':
			return costSpace
		case r == '\t':
			return costTab
		case r == '\n':
			return costNewline
		}
		return costASCIIPunct
	case unicode.Is(unicode.Han, r), unicode.Is(unicode.Hiragana, r), unicode.Is(unicode.Katakana, r), unicode.Is(unicode.Hangul, r):
		return costCJK
	case unicode.IsLetter(r) && r < 0x250:
		return costLatinExt
	case unicode.IsLetter(r) || unicode.IsDigit(r):
		return costOtherLetter
	}
	return costOther
}

func estimateTokens(s string) int {
	n := 0
	for _, r := range s {
		n += tokenCost(r)
	}
	return (n + 999) / 1000
}

type atom struct {
	text       string
	cost       int  // thousandths of a token
	breakAfter bool // a paragraph boundary follows: the preferred place to cut
}

// splitText cuts text into chunks of about targetTokens estimated tokens.
// Chunks end at paragraph boundaries when possible, then at line ends, then
// inside an over-long line at a sentence or word boundary. Fenced code blocks
// are kept together unless they alone exceed the budget. Consecutive chunks
// overlap by about overlapTokens. At most max chunks are returned (max <= 0
// means no limit); truncated reports that more text remained.
func splitText(text string, max int) (chunks []string, truncated bool) {
	atoms := toAtoms(text)
	if len(atoms) == 0 {
		return nil, false
	}
	prefix := make([]int, len(atoms)+1)
	for i, a := range atoms {
		prefix[i+1] = prefix[i] + a.cost
	}
	target := targetTokens * 1000
	overlap := overlapTokens * 1000

	start, prevEnd := 0, 0
	for start < len(atoms) {
		if max > 0 && len(chunks) >= max {
			return chunks, true
		}
		end := start
		for end < len(atoms) && (end <= prevEnd || prefix[end+1]-prefix[start] <= target) {
			end++
		}
		if end < len(atoms) {
			// Prefer the last paragraph boundary if the chunk stays at least half full.
			for i := end - 1; i > prevEnd && i >= start; i-- {
				if atoms[i-1].breakAfter && prefix[i]-prefix[start] >= target/2 {
					end = i
					break
				}
			}
		}
		chunks = append(chunks, join(atoms[start:end]))
		if end >= len(atoms) {
			break
		}
		next, ot := end, 0
		for next-1 > start && ot+atoms[next-1].cost <= overlap {
			next--
			ot += atoms[next].cost
		}
		prevEnd, start = end, next
	}
	return chunks, false
}

func join(as []atom) string {
	var b strings.Builder
	for i, a := range as {
		b.WriteString(a.text)
		if i == len(as)-1 {
			break
		}
		b.WriteByte('\n')
		if a.breakAfter {
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// toAtoms turns the document into lines tagged with paragraph boundaries.
func toAtoms(text string) []atom {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	text = strings.ReplaceAll(text, "\f", "\n\n")
	var atoms []atom
	inFence, fenceMark := false, ""
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimRightFunc(line, unicode.IsSpace)
		trimmed := strings.TrimLeft(line, " \t")
		if mark := fenceOf(trimmed); mark != "" {
			switch {
			case !inFence:
				inFence, fenceMark = true, mark
			case strings.HasPrefix(trimmed, fenceMark):
				inFence = false
			}
		}
		if trimmed == "" {
			if !inFence && len(atoms) > 0 {
				atoms[len(atoms)-1].breakAfter = true
			} else if inFence {
				atoms = append(atoms, atom{text: "", cost: costNewline})
			}
			continue
		}
		for _, piece := range splitLongLine(line) {
			atoms = append(atoms, atom{text: piece, cost: cost(piece) + costNewline})
		}
	}
	// A trailing empty fence line adds nothing.
	for len(atoms) > 0 && atoms[len(atoms)-1].text == "" {
		atoms = atoms[:len(atoms)-1]
	}
	if len(atoms) > 0 {
		atoms[len(atoms)-1].breakAfter = false
	}
	return atoms
}

func fenceOf(trimmed string) string {
	for _, m := range []string{"```", "~~~"} {
		if strings.HasPrefix(trimmed, m) {
			return m
		}
	}
	return ""
}

func cost(s string) int {
	n := 0
	for _, r := range s {
		n += tokenCost(r)
	}
	return n
}

// splitLongLine breaks a line above the chunk budget into pieces at sentence
// ends, then at spaces, and as a last resort at rune boundaries.
func splitLongLine(line string) []string {
	limit := targetTokens * 1000
	if cost(line) <= limit {
		return []string{line}
	}
	var out []string
	rest := line
	for cost(rest) > limit {
		cut := cutPoint(rest, limit)
		out = append(out, strings.TrimRight(rest[:cut], " \t"))
		rest = strings.TrimLeft(rest[cut:], " \t")
	}
	if rest != "" {
		out = append(out, rest)
	}
	return out
}

// cutPoint returns a byte offset <= the budget, preferring the last sentence
// end, then the last space, in the second half of the budget.
func cutPoint(s string, limit int) int {
	acc, hard, sentence, space := 0, 0, 0, 0
	for i, r := range s {
		acc += tokenCost(r)
		if acc > limit {
			break
		}
		hard = i + utf8.RuneLen(r)
		if acc >= limit/2 {
			switch r {
			case '.', '!', '?', '。', '！', '？', ';':
				sentence = hard
			case ' ', '\t':
				space = hard
			}
		}
	}
	switch {
	case sentence > 0:
		return sentence
	case space > 0:
		return space
	case hard > 0:
		return hard
	}
	_, size := utf8.DecodeRuneInString(s)
	return size
}
