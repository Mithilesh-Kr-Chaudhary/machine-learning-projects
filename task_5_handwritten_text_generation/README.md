# Task 5 — Handwritten Text Generation with a Character-Level RNN

## Objective

Train a character-level recurrent neural network (an LSTM) to learn patterns in text and generate new character sequences. The optional renderer then composes the generated text from real handwritten letter images.

## Important note about the supplied `english.csv`

The supplied file has 3,410 rows and two columns: `image` and `label`. It is an index of **isolated character images** (digits and letters), rather than sentences or word sequences. Therefore it cannot teach an RNN grammar, spelling, or meaningful text by itself: its rows are grouped by character label. This solution correctly uses a text corpus for RNN language learning and reserves `english.csv` for optional handwritten-image rendering.

## Files

| File | Purpose |
|---|---|
| `train_rnn.py` | Builds and trains the character-level LSTM. |
| `generate_text.py` | Samples new text from the trained model. |
| `render_handwriting.py` | Optionally renders uppercase output from image glyphs indexed by `english.csv`. |
| `corpus.txt` | Small demonstration corpus. Replace it with a larger plain-text corpus for stronger results. |
| `requirements.txt` | Python packages required to run the task. |

## Steps to run

1. Open a terminal in this folder and create an environment (optional but recommended):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the libraries:

   ```powershell
   pip install -r requirements.txt
   ```

3. Put a suitable plain-text training file in this folder. `corpus.txt` is included for a quick test. For a proper submission, use a longer text corpus in the same writing style you want to imitate. Keep it uppercase if you plan to use the supplied uppercase handwriting images for rendering.

4. Train the RNN:

   ```powershell
   python train_rnn.py --text_file corpus.txt --epochs 40 --sequence_length 80
   ```

   This creates `model/char_rnn.pt` (learned weights) and `model/vocabulary.json` (character mapping). The console loss should generally decrease as training progresses.

5. Generate new text:

   ```powershell
   python generate_text.py --model_dir model --seed_text "THE " --length 300 --temperature 0.75 --output generated_text.txt
   ```

   Lower temperature (for example `0.4`) is safer and more repetitive. Higher temperature (for example `1.0`) is more varied but may produce more mistakes.

6. Optional — create a handwritten-looking PNG. Download or locate the `Img` directory that corresponds to `english.csv`, then run:

   ```powershell
   python render_handwriting.py --text_file generated_text.txt --csv "C:\Users\eshak\Downloads\english.csv" --image_root "C:\path\to\Img" --output handwritten_output.png
   ```

   The image directory was not included next to the provided CSV during preparation, so this step needs the actual character image files. The renderer supports the CSV’s letters/digits, spaces, and new lines; unsupported punctuation is skipped.

## How the model works

1. Every unique character in the corpus becomes an integer token.
2. Overlapping windows of 80 characters are the inputs; the target is each next character.
3. The embedding layer converts token IDs into learned vectors.
4. Two LSTM layers retain context across the window and output probabilities for the next character.
5. Cross-entropy loss compares predicted probabilities with the true next characters. AdamW updates the weights, while gradient clipping keeps training stable.
6. During generation, the model predicts one character at a time. Each sampled character is fed back into the LSTM to produce the next one.

## Expected result and evaluation

With the included tiny corpus, the model is only a pipeline demonstration and will tend to memorize phrases. A much larger corpus (at least tens of thousands of characters) will yield more coherent results. Report the final training loss, include several seed/temperature samples, and describe whether spelling, spacing, and style improve as the corpus and epochs increase.
