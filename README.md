# Aya

An AI-powered intelligent wordlist generator for security research using LoRA fine-tuning with specialized datasets.

## ⚠️ Disclaimer

This tool is a PoC and designed for **authorized penetration testing, security research, and educational purposes only**. Users are responsible for ensuring compliance with applicable laws and regulations. Do not use this tool in real cases, please.

## 📋 Description

Aya is an PoC of AI system that generates intelligent and personalized wordlists for security testing. By analyzing personal information patterns and leveraging specialized datasets, Aya creates targeted password dictionaries that significantly improve the efficiency of authorized penetration testing scenarios.

## 🛠️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Personal      │───▶│   AI Pattern     │───▶│   Intelligent   │
│   Information   │    │   Analysis       │    │   Wordlist      │
│   Input         │    │   (LoRA Model)   │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                                               │
        │               ┌──────────────────┐           │
        └──────────────▶│   Leak Dataset   │───────────┘
                        │   (Preprocessed) │
                        └──────────────────┘
```

## 🔧 Input Examples

Aya can process various types of personal information:

```json
{
  "name": "John Smith",
  "birth_date": "1990-05-15",
  "location": "New York",
  "email": "john.smith@company.com",
  "phone": "+1-555-123-4567",
  "company": "TechCorp",
  "pets": ["Max", "Buddy"],
  "interests": ["football", "gaming"],
  "family": ["Sarah", "Mike"]
}
```

## 📦 Installation
