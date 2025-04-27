# Google Search Operators: The Complete List

Below is a comprehensive list of Google search operators, categorized by functionality and current reliability.

- Also Try out the Google Hacking Database 😜

## Categories

- **Working** – Fully functional and supported by Google.
- **Unreliable** – Not officially deprecated, but inconsistent in results.
- **Not Working** – Officially deprecated by Google and no longer supported.

---

## Working Operators

| Operator        | Description | Example |
|----------------|-------------|---------|
| `" "`          | Search for an exact word or phrase. | `"steve jobs"` |
| `OR`           | Search for results related to either term. | `jobs OR gates` |
| `\|`           | Same as OR. | `jobs \| gates` |
| `AND`          | Search for results containing both terms. | `jobs AND gates` |
| `-`            | Exclude a word or phrase. | `jobs -apple` |
| `*`            | Acts as a wildcard for any word. | `steve * apple` |
| `( )`          | Group multiple terms or operators. | `(ipad OR iphone) apple` |
| `define:`      | Retrieve the definition of a word or phrase. | `define:entrepreneur` |
| `cache:`       | Display the cached version of a web page. | `cache:apple.com` |
| `filetype:`    | Search for specific file types. | `apple filetype:pdf` |
| `ext:`         | Alias for `filetype:`. | `apple ext:pdf` |
| `site:`        | Limit results to a specific domain. | `site:apple.com` |
| `related:`     | Find sites related to a specified domain. | `related:apple.com` |
| `intitle:`     | Search for pages with a word in the title. | `intitle:apple` |
| `allintitle:`  | Search for pages with multiple words in the title. | `allintitle:apple iphone` |
| `inurl:`       | Search for pages with a word in the URL. | `inurl:apple` |
| `allinurl:`    | Search for pages with multiple words in the URL. | `allinurl:apple iphone` |
| `intext:`      | Search for pages with a word in the content. | `intext:apple iphone` |
| `allintext:`   | Search for pages with multiple words in the content. | `allintext:apple iphone` |
| `weather:`     | Retrieve weather information for a location. | `weather:san francisco` |
| `stocks:`      | Retrieve stock information by ticker symbol. | `stocks:aapl` |
| `map:`         | Force Google to display map results. | `map:silicon valley` |
| `movie:`       | Search for information about a movie. | `movie:steve jobs` |
| `in`           | Convert between units or currencies. | `$329 in GBP` |
| `source:`      | Filter Google News results by source. | `apple source:the_verge` |
| `before:`      | Search for results before a specified date. | `apple before:2007-06-29` |
| `after:`       | Search for results after a specified date. | `apple after:2007-06-29` |

Note: The `_` (underscore) character functions as a wildcard in Google Autocomplete but not in regular search.

---

## Unreliable Operators

| Operator        | Description | Example |
|----------------|-------------|---------|
| `#..#`          | Search within a numeric range. | `iphone case $50..$60` |
| `inanchor:`     | Find pages with specific anchor text in backlinks. | `inanchor:apple` |
| `allinanchor:`  | Similar to `inanchor:`, but for multiple words. | `allinanchor:apple iphone` |
| `AROUND(X)`     | Find terms appearing within X words of each other. | `apple AROUND(4) iphone` |
| `loc:`          | Restrict results to a specific location. | `loc:"san francisco" apple` |
| `location:`     | Restrict Google News results by location. | `location:"san francisco" apple` |
| `daterange:`    | Search within a Julian date range. | `daterange:11278-13278` |

---

## Deprecated (Not Working) Operators

| Operator        | Description | Example |
|----------------|-------------|---------|
| `~`             | Search for synonyms. Deprecated in 2013. | `~apple` |
| `+`             | Enforce exact-match for words. Deprecated in 2011. | `jobs +apple` |
| `inpostauthor:` | Search blog posts by author (Google Blog Search). | `inpostauthor:"steve jobs"` |
| `allinpostauthor:` | Same as above without quotes. | `allinpostauthor:steve jobs` |
| `inposttitle:`  | Search blog post titles. | `inposttitle:apple iphone` |
| `link:`         | Search for pages linking to a URL. Deprecated in 2017. | `link:apple.com` |
| `info:`         | Display information about a URL. Deprecated in 2017. | `info:apple.com` |
| `id:`           | Alias for `info:`. | `id:apple.com` |
| `phonebook:`    | Look up phone numbers. Deprecated in 2010. | `phonebook:tim cook` |
| `#`             | Search for Google+ hashtags. Deprecated in 2019. | `#apple` |


---





# Dorking on GitHub

## 1. Look for Exposed API Keys in Any Code File

```py
site:github.com inurl:"/blob/" "api_key" OR "apiKey"
```
- `/blob/` is a strong indicator you're inside a GitHub code file, not a repo homepage.

## 2. Raw Format Leaks (Indexed by Google)
```py
site:githubusercontent.com "api_key=" OR "SECRET_KEY="
```

- `githubusercontent.com` is the domain GitHub uses to serve raw code files.
- This hits raw code directly, not GitHub’s web interface.


## 3. Search for Firebase or Google API Keys
```py
site:github.com inurl:"/blob/" "AIzaSy"
```

## 4. Search for Tokens or Secrets
```py
site:github.com inurl:"/blob/" "token=" OR "auth_token=" OR "secret="
```

## 5. API Keys from Only Python projects
```py
site:github.com inurl:"/blob/" "api_key=" filetype:py
```

- **Or only .env secrets:**
```py
site:github.com inurl:"/blob/" "DB_PASSWORD=" filetype:env
```

## 6. LinkedIn 
```py
site:linkedin.com intitle:afiniti "Data"
```

## 7. Confidential Files
```py
filetype:pdf intitle:"Confidential" site: edu.pk
```

## 8. Passwords
```py
intitle: "index of" password
```


```py
intitle:"index of " "*.passwords.txt"
```


## 10. Exposed Webcams
```py
"WebcampXP"
```

## 11. DB Password
```py
db_password filetype:env
```


