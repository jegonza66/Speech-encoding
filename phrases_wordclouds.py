import matplotlib.pyplot as plt, pandas as pd, os
from wordcloud import WordCloud
sessions = [21, 22, 23, 24, 25, 26, 27, 29, 30]
phrases_path = os.path.normpath('Datos/phrases')

# Get all strings from phrases
total_words = ''
for session in sessions:
    print(f'Retriving words from session {session}')
    session_path = os.path.join(phrases_path,f's{session}/')
    words_session = ''
    for file in  os.listdir(session_path):
        data = pd.read_table(filepath_or_buffer=os.path.join(session_path, file),
                                header=None, 
                                sep="\t")
        phrases = [phrase.replace('-', '').replace('?', '') for phrase in data[2].tolist() if phrase != '#' and '>' not in phrase]
        words_session += ' '.join(phrases)
    total_words += words_session

# Make wordcloud
wordcloud = WordCloud(width = 1000, height = 500).generate(total_words)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, aspect='auto')
plt.axis("off")
# plt.savefig("your_file_name"+".png", bbox_inches='tight')
plt.show()
plt.close()
