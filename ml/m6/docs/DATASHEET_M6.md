# Datasheet: PawConnect AI, M6 scam detection dataset (v1)

**Owner:** Mohamed Aziz Nciri (M6) · **Date:** 17 October 2026 · **File:** `m6_dataset_v1.csv` (399 rows)

## Purpose
Train and tune M6, which flags scam messages and listing descriptions on PawConnect AI, a lost & found and pet adoption platform for Tunisia. M6 **warns users; it never blocks or bans**. No public dataset covers pet-related scams in Tunisian languages, so this one was built for the project.

## How it was made
- **Source:** written by an LLM (Claude), in batches of 100 (B01-B04), from the team's taxonomy and a written brief of Tunisian realism rules (DT/"mille", D17, Flouci, e-Dinar, mandat minute, phone-credit recharge, local places and pet names). No real messages were collected or copied.
- **Privacy:** no real people, phone numbers or links. They are replaced by `[PHONE]`, `[URL]` and `[EMAIL]`. The same replacement must be applied to real messages before scoring.
- **Review:** one strict review pass, also by the LLM. It made text, label, language and `template_id` fixes, each recorded in the `changed` column. One near-duplicate (B03-002) was removed.
- **Not yet done:** review by a native Tunisian speaker, and a two-annotator agreement check (Cohen's kappa).

## Columns
`id`, `text`, `channel` (message | listing), `language`, `is_scam` (0/1), `labels` (pipe-separated), `difficulty` (easy | hard_negative | obfuscated), `template_id`, `split`, `changed`.

## Labels (multi-label)
| Label | Definition |
|---|---|
| upfront_payment | Money demanded **before** a lost/found pet is returned or handed over |
| fake_fee | Invented fee (adoption, vaccine, transport, deposit, "premium", fine) demanded in advance |
| off_platform | Pushes the user to leave PawConnect **exclusively** ("only WhatsApp", "don't reply here") |
| phishing_link | Link or request to log in or enter data, or a request for card, OTP code, password, ID or RIB |
| impersonation | Claims to be a shelter, vet, PawConnect staff or an authority **in order to** obtain money or data |
| pressure | Invented urgency or threat to force a quick decision |
| legit | Everything else; exclusive (never combined with another label) |

Key rule: money is not a scam by itself. The test is who pays whom and when. A reward paid **after** the return, or a shelter fee paid **on site**, is legit.

## Composition
| | Rows |
|---|---|
| Scam / legit | 151 (38%) / 248 (62%) |
| Hard negatives (legit that looks suspicious) | 100 |
| Obfuscated scams (f l o u s, p@yer, numbers in words, tatweel) | 40 |
| Language | Darija Latin 139, French 80, Darija Arabic 60, Mixed 60, MSA 40, English 20 |
| Channel | 326 messages, 73 listings |

## Split
80/20 train/validation, **grouped by `template_id`**, so all variants of a scenario (including its translations) are on the same side. Train has 319 rows (101 templates) and validation has 80 rows (29 templates). Every label, language and difficulty appears in both.
**There is no test split in this file.** The frozen test set must be human-written by the team.

## Known limitations
- **Fully synthetic and LLM-generated.** The model may learn the LLM's writing style rather than real scam patterns. Validation scores will likely be **optimistic**; only the human-written test set gives a trustworthy result.
- **Small:** 399 rows, against a target of 1,500-2,000. `off_platform` (20) and English (20) are thin, so per-label and per-language scores have wide uncertainty.
- **Darija quality is unverified.** It was written and reviewed by an LLM, not by a native speaker, and real Arabizi spelling varies far more than shown here.
- **Labels come from one annotator** (the LLM), with no agreement measure.
- **Guide gaps:** extortion by a fake authority is labelled `fake_fee`. Asking for a RIB *in order to pay* a reward after the return is labelled `legit`. Both rules should be added to the annotation guide.
- **Cleaner than reality:** few typos, no voice-note transcripts, no images, no conversation context. Each row is a single message.

## Intended use
Training and threshold tuning for M6, and the comparison of rules vs TF-IDF vs transformer. **Not** for final evaluation, and not for automatic bans or blocking users.
