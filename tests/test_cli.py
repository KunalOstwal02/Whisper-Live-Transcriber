import pytest
from transcriber import build_parser


def test_language_defaults_to_english():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.language == "en"


def test_language_flag_accepted():
    parser = build_parser()
    args = parser.parse_args(["--language", "fr"])
    assert args.language == "fr"


def test_model_default():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.model == "small.en"


def test_device_default():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.device == "cuda"
