|적용 기법|비교 대상|점수 변화|소요 시간의 변화|Log 링크|작성자|
|:----|:---|:---|:----|:-----|:---|
|Teacher Forcing ratio schuduling|SATRN default 10 epoch|LB 0.4950 -> 0.5575|38min/epoch -><br> 36~68min/epoch<br>(batch 사이즈 절반)|[링크](./logs/teacherFRSchedule_10E.txt)|BGCho|
|Teacher Forcing ratio schuduling|SATRN default 50 epoch|LB 0.7707 -> 0.0.7036| 10 epoch과 동일|[링크](./logs/teacherFRSchedule_50E.txt)|BGCho|
|테스트1|테스트2|테스트3|

## 한줄평 공간  
TFR scheduling :  
처음의 성능은 양호했으나... 후반의 학습 성능은 오히려 떨어저버림. 가능한 결론으로는 :  
1. 학습의 boosting이 제대로 되어버려서 과적합 문제?를 맞닥트린 것이다. 이건 original 버전의 val score랑 비교해봐야겠다.  
2. TFR가 모델의 절대적인 성능에 영향을 줘버린거다. 10epoch 일 때는 나름 TFR가 유의미하게 바뀌지만 50이 되버리면 굉장히 완만한 곡선이 되버리기 때문에 
사실상 TFR 0.3을 대변하는 결과가 나와버린걸지도? 이게 되게 그럴듯한 설명 같다.  
3. 위의 추측이 모두 틀렸다.
