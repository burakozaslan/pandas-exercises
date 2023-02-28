

##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################

df = sns.load_dataset("titanic")
df.head(10)
df.shape
df.info()
#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df["sex"].value_counts()


#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################


df["pclass"].unique()


#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################
df["pclass"].nunique()
df["parch"].nunique()



#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")


#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df.loc[df["embarked"] == "C"]



#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

df.loc[df["embarked"] != "S"]


#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df.loc[(df["age"] < 30) & (df["sex"] == "female")]
df.query("age < 30 & sex == 'female'")  # another approach

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df.loc[(df["fare"] > 500) | (df["age"] > 70)]

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

df.isna().sum()

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.drop("who", axis=1, inplace=True)
df.head()
#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################


df["deck"].mode()[0]
df["deck"] = df["deck"].fillna(df["deck"].mode()[0])

df.head()

#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

df["age"].median()
df["age"] = df["age"].fillna(df["age"].median())
df.head()
df.isna().sum()

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(["sex","pclass"]).agg({"survived":["mean","sum","count"]})

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

#def age_change(x):
 #   if x < 30:
  #      return 1
   # else:
    #    return 0

df["age_flag"] = df["age"].apply(lambda age: 1 if age < 30 else 0)
df.head()

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

tips_df = sns.load_dataset("tips")
tips_df.head(50)
tips_df.tail(50)
#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
tips_df.groupby("time").agg({"total_bill":["sum", "min", "max", "mean"]})


#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
tips_df.groupby(["time","day"]).agg({"total_bill":["sum", "min", "max", "mean"]})


#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

tips_df.loc[(tips_df["time"] == "Lunch") & (tips_df["sex"] == "Female")].groupby(["day"]).agg({"total_bill":["sum", "min", "max", "mean"],"tip":["sum", "min", "max", "mean"]})


#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

tips_df.loc[(tips_df["size"] < 3) & (tips_df["total_bill"] < 10)].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

tips_df.head()

tips_df["total_bill_tip_sum"] = tips_df["total_bill"] + tips_df["tip"]


#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################


new_df = tips_df.sort_values(by="total_bill_tip_sum", ascending=False )[0:30].reset_index()
