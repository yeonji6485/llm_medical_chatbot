## 독거노인을 위한 Ai 어시스턴트 - 프로젝트 돌봄 
# LLM 기반 의료 상담 챗봇

## 프로젝트 개요

이 프로젝트는 고령층 사용자들이 겪는 건강 상담의 불편함을 해소하기 위해, 경량화된 한국어 LLM 모델을 기반으로 **질병 상담 챗봇**을 구축한 실험입니다. HuggingFace 기반 LLaMA 계열 모델에 대해 LoRA를 적용하여 파인튜닝하고, 실제 의료 QA 데이터셋을 기반으로 평가를 수행했습니다.

---

## 주요 기능

- LLaMA + QLoRA 기반 경량화 학습
- 사용자 질문에 대한 질병/의도 예측 + 응답
- `KoBERT`, `EEVE` 모델 기반 증상 NER 및 데이터 생성
- 건강보험 통계 기반 질병 빈도 가중치 반영
- 챗봇 인터페이스 및 CLI 실험 환경 구현

---

## 폴더 및 파일 구조

```
01_llm_medical_chatbot/
├── train_lora.ipynb          # 학습 노트북 (LoRA 적용)
├── eval_lora.ipynb           # 평가 노트북 (Trainer.evaluate)
├── train.py                  # 학습 스크립트
├── chatbot/
│   ├── chatbot.py            # KoBERT + LLaMA 통합 챗봇 클래스
│   └── run_chatbot.py        # 챗봇 CLI 실행기
├── utils/
│   ├── average.py            # 학습 로그 평균 계산
│   └── csv_editor.py         # 라벨 수동 편집기
├── evaluate/
│   └── compare_model.py      # 여러 모델 비교 평가
├── data/
│   └── train_answers.csv     # 증상-질병 QA 데이터셋
└── README.md
```

---

## 사용 기술 스택

- Python 3.10+
- Transformers, Datasets, PEFT
- bitsandbytes (8bit 추론)
- accelerate, scikit-learn

---

## 실행 환경 예시

```bash
python train.py                     # 모델 학습
python run_chatbot.py              # 챗봇 CLI 실행
```


## 결과 요약

- 학습 손실: 0.52
- 평가 정확도: 약 89.3%
- 추론 시 GPU 메모리 사용량: 약 2.8GB (8bit 기준)

---

## 향후 계획

- Layer별 selective LoRA 성능 실험
- GPTQ 기반 양자화 비교 실험
- LLaMA-3 및 Ko-LLM 시리즈 비교 추가 예정
