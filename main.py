import cv2
from pytesseract import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import keyboard


def imgupload():

    img = Image.open(r"C:\Users\fathi\Pictures\Screenshots\Screenshot (81).png")

    path_to_tesseract = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

    pytesseract.tesseract_cmd = path_to_tesseract
    ans = pytesseract.image_to_string(img)
    text = ans
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    # Creating a frequency table to keep the
    # score of each word
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(text)
    sentenceValue = dict()
    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq
    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]
    # Average value of a sentence from the original text
    average = int(sumValues / len(sentenceValue))
    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence
    print(summary)
    # Rest of your code for summarization

def pastetext():
    adil = input("paste the text here:")
    text = adil
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(text)
    # Creating a frequency table to keep the
    # score of each word
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1
    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(text)
    sentenceValue = dict()
    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq
    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]
    # Average value of a sentence from the original text
    average = int(sumValues / len(sentenceValue))
    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 *
                                                                       average)):
            summary += " " + sentence
    print(summary)

    # Rest of your code for summarization


def extract_text_from_frame(frame):
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Use pytesseract to extract text from the frame
    text = pytesseract.image_to_string(Image.fromarray(frame_rgb))
    return text

def summarize_text(text):
    # Tokenize the text into words
    words = word_tokenize(text)
    # Remove stop words from the tokenized words
    stop_words = set(stopwords.words("english"))
    words = [word.lower() for word in words if word.lower() not in stop_words]
    # Create a frequency table to keep track of word occurrences
    freq_table = {}
    for word in words:
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    # Calculate the score for each sentence based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in freq_table:
            if word in sentence.lower():
                if sentence in sentence_scores:
                    sentence_scores[sentence] += freq_table[word]
                else:
                    sentence_scores[sentence] = freq_table[word]
    # Calculate the average score of sentences
    sumValues = 0
    for sentence in sentence_scores:
        sumValues += sentence_scores[sentence]
    # Average value of a sentence from the original text

    if len(sentence_scores) > 0:  # Check if the dictionary is not empty
        average = int(sumValues / len(sentence_scores))
    else:
        average = 0

    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
        if (sentence in sentence_scores) and (sentence_scores[sentence] > (1.2 *
                                                                       average)):
            summary += " " + sentence
    return summary
    #if len(sentence_scores) > 0:
        #average_score = sum(sentence_scores.values()) / len(sentence_scores)
    #else:
        #average_score = 0
    # Generate the summary by selecting sentences with scores above the average
    #summary = ""
    #for sentence in sentence_scores:
        #if sentence_scores[sentence] > average_score:
           # summary += sentence + " "
    #return summary

def process_video():
    cap = cv2.VideoCapture(0)
    pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        text = extract_text_from_frame(frame)
        summary = summarize_text(text)
        print("Extracted Text:")
        print(text)
        print("Summary:")
        print(summary)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('q'):
            break
    cap.release()
    cv2.destroyAllWindows()







print("Press 1 for pasting text")
print("Press 2 for scanning text")
print("Press 3 for uploading an image")
n = int(input("Enter your choice: "))

if n == 1:
    pastetext()
elif n == 2:
    process_video()
elif n == 3:
    imgupload()
else:
    print("Try any other option")
