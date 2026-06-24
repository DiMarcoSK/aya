import json
import logging
import multiprocessing as mp
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .country_inferrer import CountryInferrer
from .data_validator import DataValidator
from .personal_info_extractor import PersonalInfoExtractor
from .prompt_generator import PromptGenerator


class LeakProcessor:
    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.personal_extractor = PersonalInfoExtractor()
        self.country_inferrer = CountryInferrer()
        self.prompt_generator = PromptGenerator()
        self.validator = DataValidator()
        
        self.stats = {
            'processed': 0,
            'errors': 0,
            'valid_entries': 0
        }

    def process_single_entry(self, email: str, password: str) -> dict[str, Any] | None:
        try:
            if not self.validator.validate_entry(email, password)[0]:
                return None
            
            personal_info = self.personal_extractor.extract_personal_info(email)
            country = self.country_inferrer.infer_country(email, personal_info['cleaned_name'])
            
            return {
                "prompt": self.prompt_generator.generate_detailed_prompt(email, personal_info, country),
                "response": password,
                "metadata": {
                    "email": email,
                    "personal_info": personal_info,
                    "country": country,
                    "confidence": self.country_inferrer.get_confidence_score(email, personal_info['cleaned_name'])
                }
            }
        except Exception:
            return None

    def _read_file_chunked(self, input_file: str, chunk_size: int = 10000) -> Iterator[list[str]]:
        try:
            with open(input_file, encoding='utf-8', errors='ignore') as file:
                chunk = []
                for line in file:
                    chunk.append(line.strip())
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk
        except (OSError, FileNotFoundError) as e:
            logging.error(f"Error reading file {input_file}: {e}")
            return

    def _process_chunk(self, chunk: list[str]) -> tuple[list[dict[str, Any]], int, int]:
        valid_entries = []
        processed_count = 0
        error_count = 0
        
        for line in chunk:
            processed_count += 1
            
            try:
                email, password = self.validator.clean_entry(line)
                if not email or not password:
                    error_count += 1
                    continue
                
                entry = self.process_single_entry(email, password)
                if entry:
                    valid_entries.append(entry)
                else:
                    error_count += 1
                    
            except Exception:
                error_count += 1
        
        return valid_entries, processed_count, error_count

    def process_file(self, input_file: str, output_file: str, max_entries: int | None = None, 
                    chunk_size: int = 10000, write_batch_size: int = 1000) -> None:
        
        logging.info(f"Processing leak file: {input_file}")
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as out_file:
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {}
                    batch_buffer = []
                    
                    for chunk in self._read_file_chunked(input_file, chunk_size):
                        if max_entries and self.stats['processed'] >= max_entries:
                            break
                            
                        future = executor.submit(self._process_chunk, chunk)
                        futures[future] = len(chunk)
                    
                    for future in as_completed(futures):
                        try:
                            valid_entries, processed_count, error_count = future.result()
                            
                            self.stats['processed'] += processed_count
                            self.stats['errors'] += error_count
                            self.stats['valid_entries'] += len(valid_entries)
                            
                            batch_buffer.extend(valid_entries)
                            
                            if len(batch_buffer) >= write_batch_size:
                                self._write_batch(batch_buffer, out_file)
                                batch_buffer = []
                            
                            if self.stats['processed'] % 50000 == 0:
                                logging.info(f"Processed {self.stats['processed']} entries...")
                                
                        except Exception as e:
                            logging.error(f"Error processing chunk: {e}")
                            continue
                    
                    if batch_buffer:
                        self._write_batch(batch_buffer, out_file)
            
            logging.info(f"Dataset saved to: {output_file}")
            
        except Exception as e:
            logging.error(f"Error writing output file: {e}")
            return
        
        self._log_stats()

    def _write_batch(self, batch: list[dict[str, Any]], file_handle) -> None:
        for item in batch:
            file_handle.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
        file_handle.flush()

    def process_file_sequential(self, input_file: str, output_file: str, max_entries: int | None = None) -> None:
        dataset = []
        
        logging.info(f"Processing leak file sequentially: {input_file}")
        
        try:
            with open(input_file, encoding='utf-8', errors='ignore') as file:
                for line_num, line in enumerate(file, 1):
                    if max_entries and self.stats['processed'] >= max_entries:
                        break
                    
                    try:
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
                            
                    except Exception:
                        self.stats['errors'] += 1
                        self.stats['processed'] += 1
                        continue
        
        except (OSError, FileNotFoundError) as e:
            logging.error(f"Error reading input file: {e}")
            return
        
        self._save_dataset(dataset, output_file)
        self._log_stats()

    def _save_dataset(self, dataset: list[dict], output_file: str) -> None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as out_file:
                for item in dataset:
                    out_file.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
            
            logging.info(f"Dataset saved to: {output_file}")
        except Exception as e:
            logging.error(f"Error writing output file: {e}")

    def _log_stats(self) -> None:
        logging.info("Processing completed:")
        logging.info(f"  Total processed: {self.stats['processed']}")
        logging.info(f"  Valid entries: {self.stats['valid_entries']}")
        logging.info(f"  Errors: {self.stats['errors']}")
        if self.stats['processed'] > 0:
            success_rate = (self.stats['valid_entries'] / self.stats['processed'] * 100)
            logging.info(f"  Success rate: {success_rate:.2f}%")

    def reset_stats(self) -> None:
        self.stats = {'processed': 0, 'errors': 0, 'valid_entries': 0}