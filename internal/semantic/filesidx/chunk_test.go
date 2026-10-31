package filesidx

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"unicode/utf8"
)

// realSources returns real code and prose files from this repository and the
// Go toolchain, so the chunker is exercised on text nobody tailored to it.
func realSources(t *testing.T) map[string]string {
	t.Helper()
	files := map[string]string{
		"go source":    filepath.Join("..", "..", "indexer", "indexer.go"),
		"readme":       filepath.Join("..", "..", "..", "README.md"),
		"go stdlib":    filepath.Join(runtime.GOROOT(), "src", "net", "http", "server.go"),
		"toml example": filepath.Join("..", "..", "..", "config.example.toml"),
	}
	out := map[string]string{}
	for name, p := range files {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		out[name] = string(b)
	}
	return out
}

func TestSplitTextBudgetAndCoverage(t *testing.T) {
	for name, src := range realSources(t) {
		t.Run(name, func(t *testing.T) {
			chunks, trunc := splitText(src, 0)
			if trunc || len(chunks) == 0 {
				t.Fatalf("chunks=%d truncated=%v", len(chunks), trunc)
			}
			for i, c := range chunks {
				if !utf8.ValidString(c) || strings.TrimSpace(c) == "" {
					t.Fatalf("chunk %d is empty or invalid UTF-8", i)
				}
				if n := estimateTokens(c); n > targetTokens+overlapTokens+5 {
					t.Errorf("chunk %d has ~%d tokens, limit %d", i, n, targetTokens+overlapTokens)
				}
			}
			// Nothing is lost: every line of the source is inside some chunk.
			all := strings.Join(chunks, "\n")
			for ln, line := range strings.Split(strings.ReplaceAll(src, "\r\n", "\n"), "\n") {
				if l := strings.TrimSpace(line); l != "" && !strings.Contains(all, l) {
					t.Fatalf("line %d lost: %q", ln+1, l)
				}
			}
			// Consecutive chunks overlap: the tail of one reappears in the next.
			overlapped := 0
			for i := 0; i+1 < len(chunks); i++ {
				lines := strings.Split(strings.TrimSpace(chunks[i]), "\n")
				if strings.Contains(chunks[i+1], strings.TrimSpace(lines[len(lines)-1])) {
					overlapped++
				}
			}
			if len(chunks) > 2 && overlapped < (len(chunks)-1)*3/4 {
				t.Errorf("only %d of %d boundaries overlap", overlapped, len(chunks)-1)
			}
		})
	}
}

func TestSplitTextMaxChunksKeepsTheHead(t *testing.T) {
	src := realSources(t)["go stdlib"]
	all, _ := splitText(src, 0)
	if len(all) < 10 {
		t.Fatalf("test source too small: %d chunks", len(all))
	}
	got, trunc := splitText(src, 5)
	if !trunc || len(got) != 5 {
		t.Fatalf("got %d chunks, truncated=%v", len(got), trunc)
	}
	for i := range got {
		if got[i] != all[i] {
			t.Fatalf("chunk %d differs from the unlimited split", i)
		}
	}
	if _, trunc := splitText(src, len(all)); trunc {
		t.Error("a limit equal to the chunk count must not report truncation")
	}
}

func TestSplitTextPrefersParagraphBoundaries(t *testing.T) {
	para := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 7) // ~ 90 tokens
	var paras []string
	for i := 0; i < 12; i++ {
		paras = append(paras, para)
	}
	chunks, _ := splitText(strings.Join(paras, "\n\n"), 0)
	if len(chunks) < 2 {
		t.Fatalf("%d chunks", len(chunks))
	}
	for i, c := range chunks[:len(chunks)-1] {
		if !strings.HasSuffix(strings.TrimSpace(c), "lazy dog.") {
			t.Errorf("chunk %d does not end at a paragraph end: %q", i, c[max(0, len(c)-40):])
		}
	}
}

func TestSplitTextKeepsFencedCodeTogether(t *testing.T) {
	var code strings.Builder
	code.WriteString("```go\n")
	for i := 0; i < 20; i++ {
		code.WriteString("func f() {\n\treturn\n}\n\n") // blank lines inside the fence
	}
	code.WriteString("```\n")
	prose := strings.Repeat("Some introduction that fills most of a chunk. ", 8)
	doc := prose + "\n\n" + code.String() + "\n" + prose
	chunks, _ := splitText(doc, 0)
	holder := -1
	for i, c := range chunks {
		if strings.Contains(c, "```go") {
			holder = i
		}
	}
	if holder < 0 || !strings.Contains(chunks[holder], "func f() {\n\treturn\n}\n\nfunc f()") {
		t.Fatalf("fenced block was split at an inner blank line: %q", chunks)
	}
}

func TestSplitTextLongLinesAndScripts(t *testing.T) {
	cases := map[string]string{
		"minified js":  strings.Repeat("var a=function(b){return b+1;};", 400),
		"one sentence": strings.Repeat("kelime ", 2500),
		"cjk":          strings.Repeat("这是一个用于测试的句子，它很长。", 300),
		"no spaces":    strings.Repeat("x", 9000),
		"emoji":        strings.Repeat("😀🚀", 800),
		"crlf":         strings.Repeat("satır bir\r\nsatır iki\r\n\r\n", 600),
	}
	for name, src := range cases {
		t.Run(name, func(t *testing.T) {
			chunks, _ := splitText(src, 0)
			if len(chunks) < 2 {
				t.Fatalf("a %d byte document produced %d chunks", len(src), len(chunks))
			}
			for i, c := range chunks {
				if !utf8.ValidString(c) {
					t.Fatalf("chunk %d is not valid UTF-8", i)
				}
				if n := estimateTokens(c); n > targetTokens+overlapTokens+5 {
					t.Errorf("chunk %d has ~%d tokens", i, n)
				}
			}
		})
	}
}

func TestSplitTextTrivialInputs(t *testing.T) {
	for _, in := range []string{"", "   \n\n\t\n", "\r\n"} {
		if c, trunc := splitText(in, 0); len(c) != 0 || trunc {
			t.Errorf("%q -> %q", in, c)
		}
	}
	c, _ := splitText("tek satır", 0)
	if len(c) != 1 || c[0] != "tek satır" {
		t.Errorf("short text must stay one chunk: %q", c)
	}
}

func TestSplitTextAlwaysProgresses(t *testing.T) {
	// Lines sized so that overlap plus the next line exceed the budget.
	line := strings.Repeat("a", 1500) // ~400 tokens
	var doc []string
	for i := 0; i < 40; i++ {
		doc = append(doc, line)
	}
	chunks, _ := splitText(strings.Join(doc, "\n"), 0)
	if len(chunks) < 20 || len(chunks) > 400 {
		t.Fatalf("%d chunks for 40 large lines", len(chunks))
	}
}

// TestChunkTokenBudget compares the estimate with the real tokenizer: the
// recorded real token counts of every chunk the test corpus produces must sit
// near the 512 token sweet spot and far below the model's context.
func TestChunkTokenBudget(t *testing.T) {
	tab, err := loadTable()
	if err != nil {
		t.Fatal(err)
	}
	if len(tab.Tokens) < 30 {
		t.Fatalf("only %d recorded token counts: regenerate testdata/vectors.json", len(tab.Tokens))
	}
	var sum, over int
	maxTok := 0
	for _, n := range tab.Tokens {
		sum += n
		maxTok = max(maxTok, n)
		if n > 512 {
			over++
		}
	}
	mean := sum / len(tab.Tokens)
	t.Logf("%d chunks: mean %d real tokens, max %d, %d above 512", len(tab.Tokens), mean, maxTok, over)
	if maxTok > 640 {
		t.Errorf("a chunk has %d real tokens", maxTok)
	}
	if over*10 > len(tab.Tokens) {
		t.Errorf("%d of %d chunks exceed 512 tokens", over, len(tab.Tokens))
	}
	if mean < 250 {
		t.Errorf("chunks average only %d tokens: the estimate is too pessimistic", mean)
	}
}
