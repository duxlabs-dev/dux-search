package filesidx

import (
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/text/encoding"
	"golang.org/x/text/encoding/charmap"
)

// defaultTextExts mirrors dsearch's built-in text_extensions; dsearch passes
// its configured list through Config.IsText.
var defaultTextExts = map[string]bool{
	".txt": true, ".md": true, ".go": true, ".py": true, ".js": true, ".ts": true,
	".jsx": true, ".tsx": true, ".json": true, ".yaml": true, ".yml": true,
	".toml": true, ".html": true, ".css": true, ".rs": true, ".c": true,
	".cpp": true, ".h": true, ".java": true, ".rb": true, ".php": true, ".sh": true,
}

func defaultIsText(path string) bool { return defaultTextExts[strings.ToLower(filepath.Ext(path))] }

// legacyEncodingFromLocale picks the Windows code page that older text files
// in the user's language most likely use.
func legacyEncodingFromLocale() encoding.Encoding {
	loc := ""
	for _, k := range []string{"LC_ALL", "LC_CTYPE", "LANG"} {
		if v := os.Getenv(k); v != "" && v != "C" && v != "POSIX" {
			loc = v
			break
		}
	}
	return legacyEncodingFor(loc)
}

func legacyEncodingFor(locale string) encoding.Encoding {
	lang, _, _ := strings.Cut(strings.ToLower(locale), "_")
	lang, _, _ = strings.Cut(lang, ".")
	switch lang {
	case "tr", "az":
		return charmap.Windows1254
	case "pl", "cs", "sk", "hu", "ro", "hr", "sl", "sr", "bs":
		return charmap.Windows1250
	case "ru", "uk", "bg", "be", "mk", "kk":
		return charmap.Windows1251
	case "el":
		return charmap.Windows1253
	case "he":
		return charmap.Windows1255
	case "ar", "fa", "ur":
		return charmap.Windows1256
	case "et", "lv", "lt":
		return charmap.Windows1257
	case "vi":
		return charmap.Windows1258
	}
	return charmap.Windows1252
}
