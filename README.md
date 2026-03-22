
# License Plate Recognition with TrOCR

This project uses TrOCR, a transformer-based OCR model, to read text from vehicle license plates. The pretrained model was fine-tuned on the European License Plates dataset and achieves up to 92% accuracy. It can be useful for applications such as automatic tolling, parking systems, and traffic monitoring.
## Demo
This project was deployed on Hugging Face Spaces. You can try the live demo here: [Live Demo](https://huggingface.co/spaces/mtarek123456/OCR).

Screenshot of the demo interface:
<img width="1010" height="286" alt="image" src="https://github.com/user-attachments/assets/cfb61250-cc29-4f01-a7d6-8994b3313cd7" />


```
OCR/
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── utils.py
│   └── eval.py
├── notebooks/
│   ├── 
├── main.py
├── requirements.txt
└── README.md
```

#### Acknowledgment
Sample reference image sourced from [European vehicle registration plate - Wikipedia](https://en.wikipedia.org/wiki/European_vehicle_registration_plate).
