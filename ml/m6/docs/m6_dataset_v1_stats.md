## M6 dataset v1 statistics

Rows: 399 (train 319, validation 80, 20.1% validation). Templates: 130 (train 101, validation 29). Split seed: 3399. Removed near-duplicates: B03-002.

### Scam vs legit

| | Total | Train | Validation |
|---|---|---|---|
| scam (is_scam=1) | 151 | 118 | 33 |
| legit (is_scam=0) | 248 | 201 | 47 |

### Labels (multi-label: a scam row can have several)

| | Total | Train | Validation |
|---|---|---|---|
| upfront_payment | 66 | 49 | 17 |
| fake_fee | 50 | 41 | 9 |
| off_platform | 20 | 16 | 4 |
| phishing_link | 36 | 28 | 8 |
| impersonation | 33 | 25 | 8 |
| pressure | 50 | 38 | 12 |
| legit | 248 | 201 | 47 |

### Language

| | Total | Train | Validation |
|---|---|---|---|
| darija_latin | 139 | 111 | 28 |
| french | 80 | 63 | 17 |
| darija_arabic | 60 | 50 | 10 |
| msa_arabic | 40 | 32 | 8 |
| mixed | 60 | 48 | 12 |
| english | 20 | 15 | 5 |

### Difficulty

| | Total | Train | Validation |
|---|---|---|---|
| easy | 259 | 207 | 52 |
| hard_negative | 100 | 81 | 19 |
| obfuscated | 40 | 31 | 9 |

### Channel

| | Total | Train | Validation |
|---|---|---|---|
| message | 326 | 259 | 67 |
| listing | 73 | 60 | 13 |
