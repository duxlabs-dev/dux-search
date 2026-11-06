package tokenize

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/duxlabs-dev/dux-search/internal/semantic/embed/embedtest"
)

type casesFile struct {
	Cases []struct {
		Group string  `json:"group"`
		Text  string  `json:"text"`
		IDs   []int32 `json:"ids"`
	} `json:"cases"`
}

func loadTokenizer(t testing.TB) (*Tokenizer, string) {
	t.Helper()
	dir := embedtest.ModelDir(t)
	tk, err := Load(filepath.Join(dir, "tokenizer.model"))
	if err != nil {
		t.Fatal(err)
	}
	return tk, dir
}

// TestParityWithHF compares every case with the ids the Hugging Face
// tokenizers library produced from the model's tokenizer.json
// (testdata/gen_cases.py).
func TestParityWithHF(t *testing.T) {
	tk, _ := loadTokenizer(t)
	raw, err := os.ReadFile("testdata/cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var f casesFile
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) < 200 {
		t.Fatalf("only %d cases; the acceptance bar is 200", len(f.Cases))
	}
	bad := map[string]int{}
	for i, c := range f.Cases {
		got := tk.Encode(c.Text)
		if !slices.Equal(got, c.IDs) {
			bad[c.Group]++
			n := 0
			for n < len(got) && n < len(c.IDs) && got[n] == c.IDs[n] {
				n++
			}
			t.Errorf("case %d (%s) %q: first difference at token %d (got %d tokens, want %d)", i, c.Group, trunc(c.Text), n, len(got), len(c.IDs))
		}
	}
	t.Logf("%d cases, %d differ %v", len(f.Cases), len(bad), bad)
}

func trunc(s string) string {
	if len(s) > 60 {
		return s[:60] + "..."
	}
	return s
}

// TestSpecialsMatchTokenizerJSON keeps the hard-coded added-token table equal
// to the file shipped with the model.
func TestSpecialsMatchTokenizerJSON(t *testing.T) {
	_, dir := loadTokenizer(t)
	raw, err := os.ReadFile(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Skip(err)
	}
	var tj struct {
		Added []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
			Lstrip  bool   `json:"lstrip"`
			Rstrip  bool   `json:"rstrip"`
			Norm    bool   `json:"normalized"`
			Single  bool   `json:"single_word"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(raw, &tj); err != nil {
		t.Fatal(err)
	}
	if len(tj.Added) != len(specials) {
		t.Fatalf("tokenizer.json has %d added tokens, table has %d", len(tj.Added), len(specials))
	}
	for i, a := range tj.Added {
		if a.Lstrip || a.Rstrip || a.Norm || a.Single || !a.Special {
			t.Fatalf("added token %q has flags the matcher does not implement", a.Content)
		}
		if specials[i].text != a.Content || specials[i].id != a.ID {
			t.Errorf("entry %d: table %q=%d, tokenizer.json %q=%d", i, specials[i].text, specials[i].id, a.Content, a.ID)
		}
	}
}

func TestInvalidUTF8AndEmpty(t *testing.T) {
	tk, _ := loadTokenizer(t)
	if got := tk.Encode(""); !slices.Equal(got, []int32{BOSID, EOSID}) {
		t.Fatalf("empty: %v", got)
	}
	// Invalid bytes become U+FFFD like any other unknown character, never a panic.
	bad := tk.Encode("a\xffb")
	good := tk.Encode("a�b")
	if !slices.Equal(bad, good) {
		t.Fatalf("invalid utf-8 %v != %v", bad, good)
	}
}

func TestEncodeRawHasNoSpecials(t *testing.T) {
	tk, _ := loadTokenizer(t)
	full := tk.Encode("merhaba dünya")
	raw := tk.EncodeRaw("merhaba dünya")
	if !slices.Equal(full[1:len(full)-1], raw) || full[0] != BOSID || full[len(full)-1] != EOSID {
		t.Fatalf("full %v raw %v", full, raw)
	}
}

func BenchmarkEncode(b *testing.B) {
	tk, _ := loadTokenizer(b)
	text := "task: search result | query: gözlerim yoruldu"
	for i := 0; i < b.N; i++ {
		tk.Encode(text)
	}
}

// TestParityBulk runs a large corpus when DSEARCH_TOKENIZER_BULK points to a
// JSON-lines file of {"text":..., "ids":[...]} produced with the same method
// as testdata/gen_cases.py. It is how the tokenizer was hunted for
// divergences; the committed cases are the regression set.
func TestParityBulk(t *testing.T) {
	path := os.Getenv("DSEARCH_TOKENIZER_BULK")
	if path == "" {
		t.Skip("DSEARCH_TOKENIZER_BULK not set")
	}
	tk, _ := loadTokenizer(t)
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	n, bad := 0, 0
	for dec.More() {
		var c struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
		}
		if err := dec.Decode(&c); err != nil {
			t.Fatal(err)
		}
		n++
		if got := tk.Encode(c.Text); !slices.Equal(got, c.IDs) {
			bad++
			if bad <= 20 {
				t.Errorf("%q differs", trunc(c.Text))
			}
		}
	}
	t.Logf("%d texts, %d differ", n, bad)
}
