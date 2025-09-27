# scrape_wikipedia.py

Check: Count the number of .txt files

With GLOBAL_MIN_INTERVAL = 0.25, your code allows ~4 requests/second total no matter how high --concurrency is. So --concurrency 16 ≈ --concurrency 4 in throughput.

What to change (fast + safe)

Lower the global cap

In your script, set:

```
find wikipedia_articles -type f -iname '*.txt' | wc -l
```

html to see if it returned an error. 

```
find wikipedia_articles -type f -iname '*.html' | wc -l
```

adding batch because 
i can pass 1 at a time, or supposedly 50 articles at a time (in a single request) 


Global concurrency

Start low: --concurrency 2 or --concurrency 3.

That’s 2–3 requests “in flight” at a time globally. With batch size 50, that’s up to 150 articles processed per wave.

Per-host concurrency

Keep --per-host 1.

This prevents hammering a single language Wikipedia. Each host gets one request at a time.

Batch size

Use --batch 50. Wikimedia caps at 50 titles for normal accounts; higher numbers just get split.

### Python CLI Flags

#### `--output PATH`
- **Description**: Root directory where scraped results are written.  
- **Default**: `~/Desktop/Wiki_Scrape`  
- **In the sbatch script**: overridden to `$PWD/wikipedia_articles` (the folder where the job was submitted).  

**Examples:**
```bash
--output /scratch/$USER/wiki_run
```

#### `--concurrency N`
- **Description**: Global maximum number of HTTP requests in flight across *all* Wikipedia languages.  
- **Effect**: Higher = faster, but can stress the network or hit API limits.  

**Examples:**
```bash
--concurrency 8    # conservative (slower, polite)
--concurrency 24   # aggressive (faster, more risky)
```

#### `--per-host N`
- **Description**: Maximum number of concurrent requests to a *single* Wikipedia host (e.g., `en.wikipedia.org`).  
- **Effect**: Prevents hammering one language while still allowing concurrency across others.  
- **Typical values**: 3–6  
- **Rule of thumb**: keep `concurrency ≥ 3 × per-host`.

🚦 Why keep concurrency ≥ 3 × per-host

If concurrency is too close to per-host, then one host can easily dominate the global pool.
Example:


**Examples:**
```bash
--per-host 4
--per-host 2   # stricter, good for testing
```

---

### `--max-depth D`
- **Description**: Category recursion depth.  
  - `1` → just root category pages.  
  - `2` → root + immediate subcategories.  
  - `3–5` → deeper, grows output size exponentially.  
- **Effect**: The main driver of runtime and output size.  

**Examples:**
```bash
--max-depth 1   # quick sanity run
--max-depth 3   # balanced crawl
--max-depth 5   # deep, large crawl
```

---

### `--ascii-filenames`
- **Description**: Transliterates folder/file names to ASCII (using `unidecode`).  
- **Effect**: Safer on Windows/ZIP transfers, avoids issues with non-ASCII filenames.  
- **Note**: Text content is still saved as UTF-8 inside the `.txt` files.  

**Examples:**
```bash
--ascii-filenames
```

---

# Usage Examples

### Balanced crawl
```bash
srun python wikipedia_scraper.py   --output /scratch/$USER/wiki_test   --concurrency 16   --per-host 5   --max-depth 3   --ascii-filenames
```

### Quick test (small, fast, polite)
```bash
python wikipedia_scraper.py   --output ~/Desktop/Wiki_Test   --concurrency 8   --per-host 3   --max-depth 1   --ascii-filenames
```

### Heavy run (lots of data, cluster only)
```bash
srun python wikipedia_scraper.py   --output /scratch/$USER/wiki_full   --concurrency 24   --per-host 6   --max-depth 5
```




--concurrency 16

Global limit: the maximum number of HTTP requests the scraper will have “in flight” at once across all wikis.

In plain terms: it can talk to up to 16 different Wikipedia servers/pages simultaneously.

Higher = faster, but risks hitting API rate limits or stressing your network.

--per-host 5

Per-host limit: the maximum requests allowed at once to a single Wikipedia domain (like en.wikipedia.org).

So with --concurrency 16 --per-host 5:

Total across all languages: up to 16 requests at once.

For a single wiki (say English), no more than 5 at the same time.

This prevents hammering one language’s API server.

--concurrency is global and --per-host fences per language host (e.g., en.wikipedia.org, fr.wikipedia.org). To actually use all 16 lanes, you need enough different hosts in flight

-------------

So if you don’t specify `--langs`, it runs with `--langs all` by default.

That means for every article in your root categories, the scraper will:

Save the English version (or whatever the base language is).

Use the langlinks API to discover all interlanguage links for that article.

Save every variant (French, Spanish, German, Japanese, etc.) if a link exists.

⚠️ A nuance:

“All” here means all languages linked via Wikipedia’s interlanguage links.

If a wiki doesn’t have a langlink for that article, the scraper won’t magically discover it. (That’s just how Wikipedia organizes equivalences between pages.)




-------------
Configuration Options
--output PATH

What it does: Sets the root directory where scraped results are written.

Default: ~/Desktop/Wiki_Scrape

In your batch script: overridden to $PWD/wikipedia_articles (the folder where the job was submitted).

Examples:


-----

It’s not multi-process or multi-threaded CPU parallelism. Python’s GIL means CPU-heavy work isn’t sped up; but this scraper is network-bound, so async concurrency is the right tool.




---

### Directory Map 

This is a directory map of the output from `wikipedia_scraper.py`.

```
Wiki_Scrape/
  <lang-code>/                                # e.g., en, fr, es, ru, zh, ...
    <Localized Root Category>/                # “History of ideologies”, etc., localized via langlinks
      <Subcategory 1>/
        <Sub-subcategory .../>                # recurses until --max-depth (default 4)
          <Article Title A>/                  # each *article* gets its own folder
            <LanguageName>.txt                # one .txt per language variant (base + langlinks)
            <LanguageName>.txt
            ...
          <Article Title B>/
            <LanguageName>.txt
            ...
      <Subcategory 2>/
        ...
  <another-lang-code>/                        # note: it is unlikely that a category page would have another lang code
    <Localized Root Category>/
      ...
```

-----


The script’s default is Unicode filenames (ASCII_FILENAMES=False), so without the flag it will keep non-ASCII names.

Notes

This only affects folder/file names, not the article text. The .txt contents are saved in UTF-8 either way.




Yes. The scraper talks to Wikipedia using the Unicode page/category titles, so it will fetch pages just fine even when the names contain non-ASCII characters.

What the --ascii-filenames flag changes is only the names on disk (folders/files). It transliterates those paths to ASCII so your filesystem is “safe,” but the HTTP requests still use the original Unicode titles.
