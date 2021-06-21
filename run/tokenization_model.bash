#!/bin/bash
python -m src.tasks.tokenization_model
mv tokenization.model models
mv tokenization.vocab models