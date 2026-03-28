import functions_framework
import math
from flask import jsonify

coeficientes = [-0.00234534, -0.0151067, 1.38588731, 1.36832242, -2.16535348, -0.8794301, -0.56239811, -0.58691305]
intercepto = 9.71123274

@functions_framework.http
def mlriscocredito(request):
    request_json = request.get_json(silent=True)
    request_args = request.args
    print(f"request_args: {request_args}")
    try:
        param_renda = float(request_args.get('renda'))
        param_idade = float(request_args.get('idade'))
        param_etnia = float(request_args.get('etnia'))
        param_sexo = float(request_args.get('sexo'))
        param_casapropria = float(request_args.get('casapropria'))
        param_outrasrendas = float(request_args.get('outrasrendas'))
        param_estadocivil = float(request_args.get('estadocivil'))
        param_escolaridade = float(request_args.get('escolaridade'))
        
        entrada = [param_renda, param_idade, param_etnia, param_sexo, param_casapropria, param_outrasrendas, param_estadocivil, param_escolaridade]
        


        z = intercepto + sum(entrada[i] * coeficientes[i] for i in range(len(entrada)))

        # Calcula a probabilidade usando a função sigmóide e classifica de acordo com a probabilidade
        probabilidade = 1 / (1 + math.exp(-z))
        classificacao = 1 if probabilidade >= 0.5 else 0

        mensagem_retorno = {"probabilidade": round(probabilidade, 4), "classificacao": round( classificacao, 4)}


        return jsonify(mensagem_retorno), 200

    except Exception as err:
        return jsonify({"Erro": str(err)}), 400
