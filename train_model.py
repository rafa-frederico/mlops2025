import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
import joblib
import os


def train_models(mydf: pd.DataFrame,numero_arvores : int,  altura_classificador : int, altura_regressao : int ):
    currentpath = os.path.dirname(os.path.abspath(__file__))

    # Divide os dados em X e y, e posteriormente 80% em treino e 20% em teste
    independentcols = ['renda', 'idade', 'etnia', 'sexo', 'casapropria', 'outrasrendas', 'estadocivil', 'escolaridade']
    X = mydf[independentcols]
    y = mydf['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cria o Classifier (modelo 1)
    independentcols_m1 = independentcols.copy()
    clf = rfc(n_estimators= numero_arvores, max_depth= altura_classificador)
    clf.fit(X=X_train[independentcols_m1], y=y_train)
    clf.independentcols = independentcols_m1
    clf_acuracia = clf.score(X=X_test[independentcols_m1], y=y_test)
    print("Modelo 01 (classificador), criado com acurácia de: [{0}]".format(clf_acuracia))

    # Cria o Regressor (modelo 2)
    independentcols_m2 = independentcols.copy()
    independentcols_m2.remove('etnia')
    rgs = rfr(n_estimators= numero_arvores, max_depth= altura_regressao )
    rgs.fit(X=X_train[independentcols_m2], y=y_train)
    rgs.independentcols = independentcols_m2
    rgs_acuracia = rgs.score(X=X_test[independentcols_m2], y=y_test)
    print("Modelo 02 (Regressor), criado com acurácia de: [{0}]".format(rgs_acuracia))

    # Salva ambos os modelos
    joblib.dump(clf, f'{currentpath}/models/modelo01.joblib')
    print("Modelo 01 (classificador) salvo com sucesso.")
    joblib.dump(rgs, f'{currentpath}/models/modelo02.joblib')
    print("Modelo 02 (regressor) salvo com sucesso.")
    pass


if __name__ == "__main__":
    # Carrega os dados

    mydf = pd.read_csv('./datasets/BaseDefault01.csv')
    train_models(mydf=mydf, numero_arvores=20, altura_classificador=10, altura_regressao=10)
