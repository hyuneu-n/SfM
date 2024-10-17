### 매칭 기준 수정

```
matches_good = [m1 for m1, m2 in matches if m1.distance < 0.75 * m2.distance]
```
## origin
![alt text](sfm/data/img/image.png)

## 0.95 -> 0.75
![alt text](<sfm/data/img/image (2).png>)
![alt text](<sfm/data/img/image (1).png>)