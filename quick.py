try:
    with open('sentences.txt', 'r') as file:
        text = file.read()
        if not text:
            print("File is empty")
        else:
            # print(text)
            pass
except FileNotFoundError:
    print("Could not find sentences.txt file")
except Exception as e:
    print(f"Error reading file: {e}")

# Remove newlines, extra spaces, numbers and unusual characters
text = ''.join(c for c in text if c.isalpha() or c.isspace() or c in '.!?')
text = ' '.join(text.split())

# Remove ... sequences
text = text.replace('. . .', '')

# Split into sentences
sentences = []
current = []
for word in text.split():
    current.append(word)
    if word[-1] in '.!?':
        sentences.append(' '.join(current))
        current = []
if current:
    sentences.append(' '.join(current))

# Combine short sentences
final_sentences = []
current_sentence = ''
for sentence in sentences:
    words = sentence.split()
    if len(words) < 20:
        if current_sentence:
            current_sentence = current_sentence + ' ' + sentence
            words = current_sentence.split()
            if len(words) >= 20:
                final_sentences.append(current_sentence)
                current_sentence = ''
        else:
            current_sentence = sentence
    else:
        if current_sentence:
            sentence = current_sentence + ' ' + sentence
            current_sentence = ''
        final_sentences.append(sentence)

# Add any remaining short sentence
if current_sentence:
    if len(final_sentences) > 0:
        final_sentences[-1] = final_sentences[-1] + ' ' + current_sentence
    else:
        final_sentences.append(current_sentence)

# Sort sentences by length (longer first)
final_sentences.sort(key=lambda x: len(x.split()), reverse=True)

# Write cleaned sentences
print(final_sentences)
with open('sentences_clean.txt', 'w') as file:
    for sentence in final_sentences:
        file.write(sentence + '\n')
