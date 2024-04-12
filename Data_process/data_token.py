class Tokenizer:
    """A simple tokenizer to convert between tokens and indices."""
    
    def __init__(self, token_to_index=None, index_to_token=None):
        """
        Initializes the tokenizer.

        Args:
            token_to_index (dict, optional): A dictionary mapping tokens to their indices.
            index_to_token (dict, optional): A dictionary mapping indices back to tokens.
        """
        self.token_to_index = token_to_index or {}
        self.index_to_token = index_to_token or {}
        self.unk_token = "<unk>"  # The token representing unknown words.
        self.unk_index = 0  # Default index for unknown tokens (`<unk>`).

    def fit(self, data):
        """
        Fits the tokenizer on the data to build the mapping dictionaries.

        Args:
            data (list): The dataset containing tokens to build the mappings from.
        """
        unique_tokens = sorted(set(data))  # Ensure consistent order.
        self.token_to_index = {token: index for index, token in enumerate(unique_tokens, start=1)}
        self.index_to_token = {index: token for token, index in self.token_to_index.items()}
        self.unk_index = 0  # Reset the index for `<unk>` in case it changes.

    def transform(self, data):
        """
        Transforms the data into indices using the learned token-to-index map.

        Args:
            data (list): The data to be transformed into indices.

        Returns:
            list: A list of indices corresponding to the tokens in `data`.
        """
        return [self.token_to_index.get(item, self.unk_index) for item in data]

    def reverse_transform(self, data):
        """
        Converts indices back into tokens using the index-to-token map.

        Args:
            data (list): The indices to be converted back into tokens.

        Returns:
            list: A list of tokens corresponding to the indices in `data`.
        """
        return [self.index_to_token.get(index, self.unk_token) for index in data]


# Define a list of inquiry types.
inquiry_type_list = [
    'F', 'TC', 'NC', 'M', 'PP', 'R', 'Q', 'LC', 'CA', 'OT', 'AM', 'PG', 'E', 'PN', 'AE', 'P', 'PM', 'AR', 
    'NG', 'A', 'FI', 'L', 'null', 'TG', 'FT', 'LR', 'CO', 'CF', 'EQ', 'VR', 'ED', 'AV', 'MC', 'BL', 'SH', 'HE'
]

# Create a tokenizer for inquiry types.
inquiry_type_tokenizer = Tokenizer()
inquiry_type_tokenizer.fit(inquiry_type_list)

# Define a list of income categories.
month_income_list = [
    '5,000-10,000MXN', '10,000-20,000MXN', '3,000-5,000MXN', '20,000-50,000MXN', '1,000-3,000MXN', 'below 1,000MXN', 'above 50,000MXN'
]

# Create a tokenizer for monthly income categories.
month_income_tokenizer = Tokenizer()
month_income_tokenizer.fit(month_income_list)

