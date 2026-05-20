import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가해 training.pipeline.* 임포트를 가능하게 한다
sys.path.insert(0, str(Path(__file__).parent.parent))
