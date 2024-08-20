import collections
import pandas as pd
import re

import random
import string
import math

# Initialize counters
def initialize_counters():
    deletion_counts = collections.defaultdict(int)
    insertion_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    substitution_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    character_counts = collections.defaultdict(int)
    return deletion_counts, insertion_counts, substitution_counts, character_counts

def update_counts(gt, noise, deletion_counts, insertion_counts, substitution_counts, character_counts):
    """
    Update counts for deletions, insertions, and substitutions based on aligned text pairs.

    Parameters:
    - gt: Ground truth string with alignments.
    - noise: Noisy string with alignments.
    """
    assert len(gt) == len(noise), "Aligned text pairs must have the same length."

    n = len(gt)
    i, j = 0, 0  # Pointers for gt and noise
    last_gt_char = ''  # Track the last valid character in gt

    while i < n and j < n:
        gt_char = gt[i]
        noise_char = noise[j]

        if gt_char == '@':  # Insertion case
            insertion_counts[last_gt_char][noise_char] += 1

        elif noise_char == '@':  # Deletion case
            deletion_counts[gt_char] += 1

        elif gt_char != noise_char:  # Substitution case
            substitution_counts[gt_char][noise_char] += 1
            character_counts[gt_char] += 1  # Count this as a gt occurrence
            last_gt_char = gt_char  # Update last valid character

        else:  # Correct character
            character_counts[gt_char] += 1
            last_gt_char = gt_char  # Update last valid character

        # Increment both pointers after processing the current pair
        i += 1
        j += 1

    # Handle any remaining deletions or insertions at the end
    while i < n and gt[i] != '@':
        deletion_counts[gt[i]] += 1
        i += 1
    while j < n and noise[j] != '@':
        insertion_counts[last_gt_char][noise[j]] += 1
        j += 1


# Function to calculate character distribution
def calculate_character_distribution(character_counts):
    """
    Calculate the distribution of characters based on their counts.

    Returns:
    - character_distribution: A dictionary with the probability distribution of each character.
    """
    total_characters = sum(character_counts.values())
    character_distribution = {char: count / total_characters for char, count in sorted(character_counts.items())}
    return character_distribution

# Function to calculate conditional probabilities
def calculate_conditional_probs(deletion_counts, insertion_counts, substitution_counts, character_counts):
    """
    Calculate the conditional probabilities for each character.

    Returns:
    - conditional_probs: A dictionary with conditional probabilities for each character.
    """
    conditional_probs = {}

    for char in sorted(character_counts):
        total_count = character_counts[char]
        
        # Calculate individual probabilities for this character
        delete_prob = deletion_counts[char] / total_count if char in deletion_counts else 0
        substitute_prob = sum(substitution_counts[char].values()) / total_count if char in substitution_counts else 0
        insert_prob = sum(insertion_counts[char].values()) / total_count if char in insertion_counts else 0
        
        # Correct probability is what's left after considering deletions, substitutions, and insertions
        correct_prob = 1 - (delete_prob + substitute_prob + insert_prob)
        
        # Ensure probabilities are within valid range [0, 1]
        correct_prob = max(0, min(1, correct_prob))

        conditional_probs[char] = {
            'correct': correct_prob,
            'substitute': substitute_prob,
            'delete': delete_prob,
            'insert': insert_prob
        }

    return conditional_probs

# Function to generate substitution and insertion tables
def generate_substitution_insertion_tables(substitution_counts, insertion_counts, character_counts):
    """
    Generate the substitution and insertion tables based on observed counts.

    Returns:
    - substitution_table: A dictionary with substitution probabilities for each character.
    - insertion_table: A dictionary with insertion probabilities for each character.
    """
    substitution_table = {}
    insertion_table = {}

    for char in sorted(substitution_counts):
        total_subs = sum(substitution_counts[char].values())
        substitution_table[char] = {sub_char: count / total_subs for sub_char, count in sorted(substitution_counts[char].items())}
    
    for char in sorted(insertion_counts):
        total_ins = sum(insertion_counts[char].values())
        insertion_table[char] = {ins_char: count / total_ins for ins_char, count in sorted(insertion_counts[char].items())}
    
    return substitution_table, insertion_table

# Add default values
def add_default_values(conditional_probs, substitution_table, insertion_table, character_distribution):
    """
    Add default values for characters not explicitly listed.

    Returns:
    - Updated tables with default values.
    """
    default_conditional = { 
        'correct': sum(d['correct'] for d in conditional_probs.values()) / len(conditional_probs),
        'substitute': sum(d['substitute'] for d in conditional_probs.values()) / len(conditional_probs),
        'delete': sum(d['delete'] for d in conditional_probs.values()) / len(conditional_probs),
        'insert': sum(d['insert'] for d in conditional_probs.values()) / len(conditional_probs)
    }
    
    default_substitution = { 
        char: prob for char, prob in sorted(character_distribution.items())
    }
    
    default_insertion = {
        char: prob for char, prob in sorted(character_distribution.items())
    }
    
    conditional_probs['default'] = default_conditional
    substitution_table['default'] = default_substitution
    insertion_table['default'] = default_insertion
    
    return conditional_probs, substitution_table, insertion_table


def modify_and_renormalize_probs(conditional_probs, column, desired_value):
    """
    Modify a specific column in the conditional probability dictionary to the desired value,
    ensuring probabilities remain within [0, 1], and then renormalize so they sum to 1.

    Parameters:
    - conditional_probs: A dictionary of conditional probabilities for each character.
    - column: The column (correct, substitute, delete, insert) to modify.
    - desired_value: The desired value for the selected column.

    Returns:
    - modified_probs: A new dictionary with the modified and renormalized probabilities.
    """
    modified_probs = {}

    for char, probs in conditional_probs.items():
        # Set the selected column to the desired value
        scaled_value = max(0, min(1, desired_value))

        # Calculate the remaining total for the other columns
        remaining_total = 1 - scaled_value

        # Calculate the total of the other columns before scaling
        original_remaining_total = sum(probs[key] for key in probs if key != column)
        
        # Renormalize the other columns
        modified_probs[char] = {}
        for key in probs:
            if key == column:
                modified_probs[char][key] = scaled_value
            else:
                if original_remaining_total > 0:
                    new_value = probs[key] * remaining_total / original_remaining_total
                else:
                    new_value = 0  # Handle edge case where original_remaining_total might be zero
                
                # Ensure renormalized value is within [0, 1]
                modified_probs[char][key] = max(0, min(1, new_value))
        
        # Final adjustment to ensure all probabilities sum to 1
        total_prob = sum(modified_probs[char].values())
        if total_prob != 1:
            for key in modified_probs[char]:
                modified_probs[char][key] = modified_probs[char][key] / total_prob

    return modified_probs


def calculate_joint_probabilities(conditional_probs, character_distribution):
    """
    Calculate the joint probabilities by multiplying conditional probabilities by the character distribution
    and then sum these joint probabilities.

    Parameters:
    - conditional_probs: A dictionary of conditional probabilities for each character.
    - character_distribution: A dictionary of probability distributions for each character.

    Returns:
    - joint_probs: A dictionary with summed joint probabilities for 'correct', 'substitute', 'delete', and 'insert'.
    """
    joint_probs = {
        'correct': 0.0,
        'substitute': 0.0,
        'delete': 0.0,
        'insert': 0.0
    }
    
    # Calculate joint probabilities
    for char, cond_prob in conditional_probs.items():
        if char in character_distribution:
            char_prob = character_distribution[char]
            joint_probs['correct'] += cond_prob['correct'] * char_prob
            joint_probs['substitute'] += cond_prob['substitute'] * char_prob
            joint_probs['delete'] += cond_prob['delete'] * char_prob
            joint_probs['insert'] += cond_prob['insert'] * char_prob
    
    return joint_probs


class Character:
    def __init__(self, char):
        self.original = char
        self.current = char
        self.state = "Correct"
        self.insertions = []
class CorruptionEngine:
    def __init__(self, conditional_probs, substitution_table, insertion_table):
        self.conditional_probs = conditional_probs
        self.substitution_table = substitution_table
        self.insertion_table = insertion_table
        
        # Ensure default options exist
        if 'default' not in self.conditional_probs:
            raise ValueError("conditional_probs must include a 'default' entry")
        if 'default' not in self.substitution_table:
            raise ValueError("substitution_table must include a 'default' entry")
        if 'default' not in self.insertion_table:
            raise ValueError("insertion_table must include a 'default' entry")

    def process_character(self, char):
        error_count = 0
        while True:
            if char.state == "Correct":
                char.state = self.choose_action(char.original)
                if char.state == "Correct":
                    return self.finalize(char), error_count
            elif char.state == "Substituted":
                char.current = self.substitute(char.original)
                error_count += 1  # Count as one substitution error
                if self.choose_action(char.current) != "Inserted":
                    return self.finalize(char), error_count
                char.state = "Inserted"
            elif char.state == "Deleted":
                error_count += 1  # Count as one deletion error
                return [], error_count
            elif char.state == "Inserted":
                inserted = self.insert_character(char.current)
                char.insertions.append(inserted)
                error_count += 1  # Count each insertion as one error
                if self.choose_action(inserted) != "Inserted":
                    return self.finalize(char), error_count
            char.current = char.insertions[-1] if char.insertions else char.current

    def choose_action(self, char):
        probs = self.conditional_probs.get(char, self.conditional_probs['default'])
        return random.choices(["Correct", "Substituted", "Deleted", "Inserted"], 
                              weights=[probs['correct'], probs['substitute'], probs['delete'], probs['insert']])[0]

    def substitute(self, char):
        sub_options = self.substitution_table.get(char, self.substitution_table['default'])
        choices = list(sub_options.keys())
        weights = list(sub_options.values())
        return random.choices(choices, weights=weights)[0]

    def insert_character(self, prev_char):
        insert_options = self.insertion_table.get(prev_char, self.insertion_table['default'])
        choices = list(insert_options.keys())
        weights = list(insert_options.values())
        return random.choices(choices, weights=weights)[0]

    def finalize(self, char):
        return [char.current] + char.insertions

    def corrupt_text(self, text):
        corrupted_chars = []
        total_char_errors = 0
        total_chars = len(text)

        for char in text:
            original_char = char
            corrupted_char, error_count = self.process_character(Character(char))
            corrupted_chars.extend(corrupted_char)

            # Increment total character errors by the error count returned from process_character
            total_char_errors += error_count

        # Calculate CER
        corrupted_text = ''.join(corrupted_chars)
        cer = total_char_errors / total_chars if total_chars > 0 else 0

        return corrupted_text, cer

    
class WERBasedCorruptionEngine(CorruptionEngine):
    """ 
    Corrupts text based on a target WER and CER, and returns the corrupted text along with the actual WER and CER.
    """
    def __init__(self, conditional_probs, substitution_table, insertion_table):
        super().__init__(conditional_probs, substitution_table, insertion_table)

    def split_text(self, text):
        """
        Split the text into words, keeping punctuation and spaces as part of the words.
        This function ensures that spaces and punctuation are preserved in the corruption process.
        """
        return re.findall(r'\S+\s*', text)

    def corrupt_text_with_wer_cer(self, text, target_wer, target_cer):
        words = self.split_text(text)
        num_words = len([word for word in words if not word.isspace()])

        # Determine the number of words to corrupt based on the target WER
        num_words_to_corrupt = math.ceil(target_wer * num_words)
        words_to_corrupt_indices = random.sample(range(len(words)), num_words_to_corrupt)

        # Calculate the fraction of characters that will be corrupted
        selected_chars_count = sum(len(words[i]) for i in words_to_corrupt_indices)
        total_chars_count = len(text)
        selected_fraction = selected_chars_count / total_chars_count

        # Calculate the effective CER for the selected words and spaces
        effective_cer = target_cer / selected_fraction

        # Modify and renormalize probabilities based on the effective correct rate
        effective_correct_rate = 1 - effective_cer
        modified_conditional_probs = modify_and_renormalize_probs(self.conditional_probs, column='correct', desired_value=effective_correct_rate)

        # Initialize a new corruption engine with modified probabilities
        modified_scrambler = CorruptionEngine(modified_conditional_probs, self.substitution_table, self.insertion_table)

        # Corrupt the selected words and track errors
        corrupted_words = []
        total_char_errors = 0

        for i, word in enumerate(words):
            if i in words_to_corrupt_indices:
                corrupted_word, cer = modified_scrambler.corrupt_text(word)
                total_char_errors += cer * len(word)  # Scale the CER by the word length
            else:
                corrupted_word = word  # Leave the word uncorrupted
            corrupted_words.append(corrupted_word)

        # Calculate the actual WER and CER
        actual_wer = len(words_to_corrupt_indices) / num_words if num_words > 0 else 0
        actual_cer = total_char_errors / total_chars_count if total_chars_count > 0 else 0

        # Join the corrupted words back into a single string
        corrupted_text = ''.join(corrupted_words)

        return corrupted_text, actual_wer, actual_cer




