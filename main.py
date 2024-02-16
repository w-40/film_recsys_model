from flask import Flask, request, jsonify
from flask_cors import CORS

import ddsm

app = Flask(__name__)
CORS(app)


# 封装返回结果的函数
def success_response(data):
    return jsonify({"code": "200", "data": data, "total": len(data.get("recommendations"))})


def error_response(message):
    return jsonify({"code": "201", "message": message})


# 定义接口
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()

    # 获取电影名称
    movie_name = data.get('movieName', None)

    if movie_name:
        try:
            # 调用模型获取推荐电影
            recommendations = ddsm.get_recommendations(movie_name)
            return success_response({"recommendations": recommendations.tolist()})
        except Exception as e:
            return error_response(str(e))
    else:
        return error_response("Invalid input")


if __name__ == '__main__':
    app.run(port=5000)
