# Titanic Tahmin Projesi

Bu proje, Titanic gemisinde yolcuların hayatta kalma olasılıklarını makine öğrenmesi teknikleri kullanarak tahmin etmeyi amaçlamaktadır. Veri seti, yolcuların demografik bilgileri, bilet sınıfı ve diğer çeşitli özellikleri içermektedir. Proje, farklı modeller kullanarak bu tahminleri gerçekleştirmekte ve modellerin performansını karşılaştırmaktadır. Bu çalışma, bir Kaggle yarışmasında gerçekleştirilmiş olup, doğruluk oranı %76.794 olarak elde edilmiştir.

<a href="https://www.kaggle.com/code/mevltbaaran/titanic-prediction" target="_blank">Kaggle Proje Sayfası</a>

## İçindekiler
1. [Giriş](#giriş)
2. [Veri Seti Tanımı](#veri-seti-tanımı)
3. [Veri Ön İşleme](#veri-ön-işleme)
4. [Model Eğitimi](#model-eğitimi)
5. [Sonuç](#sonuç)
6. [Gelecek Çalışmalar](#gelecek-çalışmalar)
7. [Kaynaklar](#kaynaklar)

## Giriş

Bu projede, Titanic gemisinde yolcuların hayatta kalma olasılıklarını tahmin etmek için çeşitli makine öğrenmesi modelleri kullanılmıştır. Bu modellerin performans değerlendirmesi Kaggle tarafından yapılmış olup, yarışma sıralaması için doğruluk oranı %76.794 olarak elde edilmiştir.

## Veri Seti Tanımı

Veri seti, Titanic gemisindeki yolcuların demografik bilgileri, bilet sınıfı ve diğer çeşitli özellikleri içermektedir. Bu veriler, yolcuların hayatta kalma olasılıklarını belirlemek için kullanılmıştır.
### Dataset info
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
```

### Data Load
```
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
```

## Veri Ön İşleme

Veri setindeki eksik ve aykırı-turtarsız veriler temizlenmiş ve ölçeklendirilmiştir, aykırı değerlerin temizlenmesinde quantile yöntemi kullanılmıştır. Bu aşama, modellerin daha doğru sonuçlar vermesi için oldukça önemlidir.
```
#Passenger_ID, yolcunun hayatta kalıp kalmadığı bilgisi hakkında tahmin yürütebilceğimiz bir sütun değildir. Ayrıca, name Sütunu da kısmen ID niteliği taşıdığından bu sütun da işimize pek yaramayacaktır.
df_train2 = df_train.drop(['PassengerId', 'Name'], axis=1)
```
```
age_means = df_train2.groupby('Survived')['Age'].mean()

# NaN değerlerini grup ortalamaları ile doldurma
df_train2['Age'] = df_train2.apply(
    lambda row: age_means[row['Survived']] if pd.isna(row['Age']) else row['Age'],
    axis=1
)
```
```
# NAN Kabin verileri total verilerin yarısında fazla olduğu ve anlamlandırmadığımız için siliyoruz
df_train2 = df_train2.drop('Cabin', axis=1)
```
```
# Bilet bilgisi bizim için bir anlam ifade etmez, buna dikkat etmemiz gerekiyor
df_train2 = df_train2.drop('Ticket', axis=1)
```
```
# Kategorik verileri sayısal verilere dönüştürüyoruz
df_train2 = pd.get_dummies(df_train2, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
# drop_first=True ile kukla verileri kaldırıyoruz
```
```
outliers = df_train2[['Age', 'SibSp', 'Parch', 'Fare']].quantile(0.99)
```

## Model Eğitimi

Projede kullanılan model:
- Yapay Sinir Ağları
```
x = df_train2.drop('Survived', axis=1)
y = df_train2['Survived']
```
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Sonuç

Yapay Sinir Ağları modeli, Titanic yolcularının hayatta kalma tahmininde iyi bir performans göstermiştir. Kaggle yarışmasında %76.794 doğruluk oranı elde edilmiştir. Gelecekte daha büyük veri setleri ve daha gelişmiş modeller kullanarak tahmin doğruluğunu artırmayı planlıyoruz.

![image](https://github.com/mevlt01001/-titanic_prediction-/assets/114837266/8d1227af-8054-4b89-8964-d7ae37d974b8)



---
<a href="https://www.kaggle.com/code/mevltbaaran/titanic-prediction" target="_blank">Kaggle Proje Sayfası</a>
