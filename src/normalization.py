import re

class Normalize:
    #Function to keep English and Bangla
    def normalize_script(self, text): 
        # Keep Bangla and English characters, remove others
        text = re.sub(r"[^\u0980-\u09FFa-zA-Z\s]", "", text)
        return text

    # Function to remove emojis
    def remove_emojis(self, text): 
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F700-\U0001F77F"  # Alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric shapes extended
            "\U0001F800-\U0001F8FF"  # Supplemental arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
            "\U0001FA00-\U0001FA6F"  # Chess symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and pictographs extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"  # Enclosed characters
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    # Function to remove URLs
    def remove_urls(self, text): 
        url_pattern = re.compile(r"https?://\S+|www\.\S+")
        return url_pattern.sub(r'', text)

    # Function to remove special characters and retain Bangla text
    def remove_special_characters(self, text): 
        # Keep Bangla characters, digits, and some punctuation
        return re.sub(r"[^ঀ-৿০-৯a-zA-Z\s.,!?]", "", text)

    # Function to normalize whitespace
    def normalize_whitespace(self, text): 
        return re.sub(r"\s+", " ", text).strip()

    # Function to remove HTML tags from text
    def remove_html_tags(self,text):
        pattern = re.compile('<.*?>')
        return pattern.sub(r'', text)

    # Function to convert English digits to Bangla digits
    def convert_english_to_bangla_digits(self, text): 
        english_to_bangla_digit_map = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
        return text.translate(english_to_bangla_digit_map)

    # Function to normalize text
    def normalize_text(self, text): 
        text = self.remove_html_tags(text)  
        text = self.normalize_script(text)  
        text = self.remove_emojis(text)  
        text = self.remove_urls(text)  
        text = self.remove_special_characters(text)  
        text = self.normalize_whitespace(text)  
        text = self.convert_english_to_bangla_digits(text) 
        return text