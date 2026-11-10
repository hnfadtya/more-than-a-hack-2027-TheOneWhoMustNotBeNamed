# External labelled data (other games only)

Nothing here is committed except this file. Everything in this folder is data about **other
matches**, which RULES.md allows for training and validation.

## IDSSE: 7 German matches, DFL positional data, CC BY 4.0

Bassek, Weber, Rein & Memmert (2025), *An integrated dataset of spatiotemporal and event data in
elite soccer*, Scientific Data. Every frame has `BallStatus` and `BallPossession`.

Figshare blocks scripted downloads, so download these **in a browser** and save them as
`external/idsse/<match_id>/{meta,tracking,event}.xml`:

| match_id | match | meta | tracking | event |
|---|---|---|---|---|
| J03WMX | 1. FC Köln vs FC Bayern München | [51643475](https://springernature.figshare.com/ndownloader/files/51643475) | [51643514](https://springernature.figshare.com/ndownloader/files/51643514) | [51643493](https://springernature.figshare.com/ndownloader/files/51643493) |
| J03WN1 | VfL Bochum vs Bayer 04 Leverkusen | [51643472](https://springernature.figshare.com/ndownloader/files/51643472) | [51643517](https://springernature.figshare.com/ndownloader/files/51643517) | [51643496](https://springernature.figshare.com/ndownloader/files/51643496) |
| J03WPY | Fortuna Düsseldorf vs 1. FC Nürnberg | [51643487](https://springernature.figshare.com/ndownloader/files/51643487) | [51643526](https://springernature.figshare.com/ndownloader/files/51643526) | [51643505](https://springernature.figshare.com/ndownloader/files/51643505) |
| J03WOH | Fortuna Düsseldorf vs SSV Jahn Regensburg | [51643478](https://springernature.figshare.com/ndownloader/files/51643478) | [51643520](https://springernature.figshare.com/ndownloader/files/51643520) | [51643499](https://springernature.figshare.com/ndownloader/files/51643499) |
| J03WQQ | Fortuna Düsseldorf vs FC St. Pauli | [51643484](https://springernature.figshare.com/ndownloader/files/51643484) | [51643529](https://springernature.figshare.com/ndownloader/files/51643529) | [51643508](https://springernature.figshare.com/ndownloader/files/51643508) |
| J03WOY | Fortuna Düsseldorf vs F.C. Hansa Rostock | [51643481](https://springernature.figshare.com/ndownloader/files/51643481) | [51643523](https://springernature.figshare.com/ndownloader/files/51643523) | [51643502](https://springernature.figshare.com/ndownloader/files/51643502) |
| J03WR9 | Fortuna Düsseldorf vs 1. FC Kaiserslautern | [51643490](https://springernature.figshare.com/ndownloader/files/51643490) | [51643532](https://springernature.figshare.com/ndownloader/files/51643532) | [51643511](https://springernature.figshare.com/ndownloader/files/51643511) |

Then convert all of them (once; cached under `cache/`):

```bash
python -m mtah.external.idsse
```
