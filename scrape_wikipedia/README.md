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

So if you don’t specify --langs, it runs with --langs all by default.

That means for every article in your root categories, the scraper will:

Save the English version (or whatever the base language is).

Use the langlinks API to discover all interlanguage links for that article.

Save every variant (French, Spanish, German, Japanese, etc.) if a link exists.

⚠️ A nuance:

“All” here means all languages linked via Wikipedia’s interlanguage links.

If a wiki doesn’t have a langlink for that article, the scraper won’t magically discover it. (That’s just how Wikipedia organizes equivalences between pages.)




-------------



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
  <another-lang-code>/                        # Although it is unlikely that a category page would have another lang code
    <Localized Root Category>/
      ...
```

-----


The script’s default is Unicode filenames (ASCII_FILENAMES=False), so without the flag it will keep non-ASCII names.

Notes

This only affects folder/file names, not the article text. The .txt contents are saved in UTF-8 either way.




Yes. The scraper talks to Wikipedia using the Unicode page/category titles, so it will fetch pages just fine even when the names contain non-ASCII characters.

What the --ascii-filenames flag changes is only the names on disk (folders/files). It transliterates those paths to ASCII so your filesystem is “safe,” but the HTTP requests still use the original Unicode titles.
