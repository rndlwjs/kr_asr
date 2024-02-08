# Conformer ASR 개발<br/>

## Architecture:<br/>

## 1. Encoder 개발<br/>
![conformer](https://github.com/rndlwjs/kr_asr/assets/70250234/8732976d-a055-4e74-832b-f0d8065bbbde)<br/>
- Feed Forward Module(1)<br/>
- Multi-head self attention(2)<br/>
<!--
Transformer-XL, relative sinusoidal positional encoding scheme: It allows the self-attention module to generalize better on different input length and the resulting encoder is more robust to the variance of the utterance length.
[reference] Transformer-xl: Attentive language models beyond a
fixed-length context

relative positional encoding 구현 필요

pre-norm residual units with dropout: training and regularizing deeper models

Multi-head Attention = Scaled Dot-Product Attention 여러개 있는 것!

Scaled Dot-Product Attention 구현 필요 (Q,K,V)

Transformer (encoder+decoder) vs Multi-head attention
-->
- Convolution Module(1)<br/>
- Feed Forward Module(1)<br/>

Due date: <br/>
(1) 24/02/07 - 24/02/14
(2) 24/02/14 - 24/02/21

## 2. 데이터셋 처리<br/>
Kspoonspeech
<!--
librispeech과 유사한 1000시간 데이터 찾기
강의용 음성 / 대화형 음성 / 소아용 음성
한영 혼합
-->
## 3. Decoder 연결<br/>
<!--
LSTM decoder
CTC loss / Transducer
-->