"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from codehid.processors.base_processor import BaseProcessor
from codehid.processors.codebert_text import CodeBERTTextProcessor

from codehid.common.registry import registry

__all__ = [
    "BaseProcessor",
    "CodeBERTTextProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("codebert_text", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
