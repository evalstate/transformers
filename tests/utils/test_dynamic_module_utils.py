# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys

import pytest

from transformers.dynamic_module_utils import custom_object_save, get_imports


TOP_LEVEL_IMPORT = """
import os
"""

IMPORT_IN_FUNCTION = """
def foo():
    import os
    return False
"""

DEEPLY_NESTED_IMPORT = """
def foo():
    def bar():
        if True:
            import os
        return False
    return bar()
"""

TOP_LEVEL_TRY_IMPORT = """
import os

try:
    import bar
except ImportError:
    raise ValueError()
"""

TRY_IMPORT_IN_FUNCTION = """
import os

def foo():
    try:
        import bar
    except ImportError:
        raise ValueError()
"""

MULTIPLE_EXCEPTS_IMPORT = """
import os

try:
    import bar
except (ImportError, AttributeError):
    raise ValueError()
"""

EXCEPT_AS_IMPORT = """
import os

try:
    import bar
except ImportError as e:
    raise ValueError()
"""

GENERIC_EXCEPT_IMPORT = """
import os

try:
    import bar
except:
    raise ValueError()
"""

MULTILINE_TRY_IMPORT = """
import os

try:
    import bar
    import baz
except ImportError:
    raise ValueError()
"""

MULTILINE_BOTH_IMPORT = """
import os

try:
    import bar
    import baz
except ImportError:
    x = 1
    raise ValueError()
"""

CASES = [
    TOP_LEVEL_IMPORT,
    IMPORT_IN_FUNCTION,
    DEEPLY_NESTED_IMPORT,
    TOP_LEVEL_TRY_IMPORT,
    GENERIC_EXCEPT_IMPORT,
    MULTILINE_TRY_IMPORT,
    MULTILINE_BOTH_IMPORT,
    MULTIPLE_EXCEPTS_IMPORT,
    EXCEPT_AS_IMPORT,
    TRY_IMPORT_IN_FUNCTION,
]


@pytest.mark.parametrize("case", CASES)
def test_import_parsing(tmp_path, case):
    tmp_file_path = os.path.join(tmp_path, "test_file.py")
    with open(tmp_file_path, "w") as _tmp_file:
        _tmp_file.write(case)

    parsed_imports = get_imports(tmp_file_path)
    assert parsed_imports == ["os"]


def test_custom_object_save_destination_is_writable_when_source_is_readonly(tmp_path, monkeypatch):
    # Regression test for https://github.com/huggingface/transformers/issues/45684:
    # `custom_object_save` used `shutil.copy`, which preserves source mode bits, so
    # a read-only source (e.g. a Perforce-managed file) produced a read-only copy
    # in the saved-model directory.
    src = tmp_path / "my_custom_module.py"
    src.write_text("class CustomThing:\n    pass\n")

    spec = importlib.util.spec_from_file_location("my_custom_module", src)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "my_custom_module", module)
    spec.loader.exec_module(module)

    src.chmod(0o444)  # read-only source

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    custom_object_save(module.CustomThing, str(out_dir))

    dest = out_dir / "my_custom_module.py"
    assert dest.exists()
    assert os.access(dest, os.W_OK), f"dest mode={oct(dest.stat().st_mode)} should be writable"
