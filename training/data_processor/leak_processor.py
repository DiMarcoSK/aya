import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from .personal_info_extractor import PersonalInfoExtractor
from .country_inferrer import CountryInferrer
from .prompt_generator import PromptGenerator
from .data_validator import DataValidator


class LeakProcessor:
    """Main processor for leak datasets"""
    
    def __init__(self):
        self.personal_extractor = PersonalInfoExtractor()
        self.country_inferrer = CountryInferrer()
        self.prompt_generator = PromptGenerator()
        self.validator = DataValidator()
        
        self.stats = {
            'processed': 0,
            'errors': 0,
            'valid_entries': 0
        }

    def process_single_entry(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Process a single email:password entry"""
        is_valid, error = self.validator.validate_entry(email, password)
        if not is_valid:
            return None
        
        personal_info = self.personal_extractor.extract_personal_info(email)
        country = self.country_inferrer.infer_country(email, personal_info['cleaned_name'])
        
        prompt = self.prompt_generator.generate_detailed_prompt(email, personal_info, country)
        
        return {
            "prompt": prompt,
            "response": password,
            "metadata": {
                "email": email,
                "personal_info": personal_info,
                "country": country,
                "confidence": self.country_inferrer.get_confidence_score(email, personal_info['cleaned_name'])
            }
        }

    def process_file(self, input_file: str, output_file: str, max_entries: Optional[int] = None) -> None:
        """Process entire leak file"""
        dataset = []
        
        logging.info(f"Processing leak file: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
                for line_num, line in enumerate(file, 1):
                    if max_entries and self.stats['processed'] >= max_entries:
                        break
                    
                    email, password = self.validator.clean_entry(line)
                    if not email or not password:
                        self.stats['errors'] += 1
                        continue
                    
                    entry = self.process_single_entry(email, password)
                    if entry:
                        dataset.append(entry)
                        self.stats['valid_entries'] += 1
                    else:
                        self.stats['errors'] += 1
                    
                    self.stats['processed'] += 1
                    
                    if self.stats['processed'] % 10000 == 0:
                        logging.info(f"Processed {self.stats['processed']} entries...")
        
        except FileNotFoundError:
            logging.error(f"Input file not found: {input_file}")
            return
        except Exception as e:
            logging.error(f"Error reading input file: {e}")
            return
        
        self._save_dataset(dataset, output_file)
        self._log_stats()

    def _save_dataset(self, dataset: List[Dict], output_file: str) -> None:
        """Save dataset to JSONL file"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as out_file:
                for item in dataset:
                    out_file.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logging.info(f"Dataset saved to: {output_file}")
        except Exception as e:
            logging.error(f"Error writing output file: {e}")

    def _log_stats(self) -> None:
        """Log processing statistics"""
        logging.info(f"Processing completed:")
        logging.info(f"  Total processed: {self.stats['processed']}")
        logging.info(f"  Valid entries: {self.stats['valid_entries']}")
        logging.info(f"  Errors: {self.stats['errors']}")
        logging.info(f"  Success rate: {(self.stats['valid_entries']/self.stats['processed']*100):.2f}%")


