from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, ImageMessage, FlexSendMessage,CarouselContainer,BubbleContainer
from image_change import mosic_change, art_change, dot_change, illust_change
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

work = {}
path_w1 = 'saveid.txt'
path_w2 = 'savereply.txt'
app = Flask(__name__)

YOUR_CHANNEL_ACCESS_TOKEN = "21MB2pzMrEs0JNqAdPTPyxFJmnaljipr9bLiUuMJrPWaLeCPHmK1tnqK23FoVL9kjqnpmyaJ0jFu3/KBCKl+O0WKIYzZ6lqfNEcAGaw3ag8aOwVlNzFsgmgVjiyewGsJOjnlogELVfGGTqz/PRJimwdB04t89/1O/w1cDnyilFU="
YOUR_CHANNEL_SECRET = "b392c7fd703eba783a31c2c6cb80a890"
line_bot_api = LineBotApi(YOUR_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(YOUR_CHANNEL_SECRET)
FQDN = " https://team-hagi-project.herokuapp.com"



@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]

    body = request.get_data(as_text=True)
    print("Request body" + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    line_bot_api.reply_message(event.reply_token,
       [
           TextSendMessage(text=event.message.text),
           #TextSendMessage(text="おはよ------"),
           #TextSendMessage(text="顔、目を検知できませんでした。"),
           #TextSendMessage(text=event.message.id),
       ]
       )
    print("取得イヴェント:{}".format(event))
    print("取得イヴェントメッセージID:{}".format(event.message.id))
    print("リプライトークン：{}".format(event.reply_token))
    print("------リプライ型------")
    print(type(event.reply_token))

    #モザイク(目)
    if event.message.text == "1":
        print("通過: {}".format(event.message.text))
        with open(path_w1) as f:
            work = f.read()
        with open(path_w2) as f2:
            work1 = f2.read()
        handle_send_message(work,work1)

    #線画
    elif event.message.text == "2":
        print("通過: {}".format(event.message.text))
        with open(path_w1) as f:
            work = f.read()
        with open(path_w2) as f2:
            work1 = f2.read()
        handle_send_message2(work,work1)

    #イラスト風
    elif event.message.text == "3":
        print("通過: {}".format(event.message.text))
        with open(path_w1) as f:
            work = f.read()
        with open(path_w2) as f2:
            work1 = f2.read()
        handle_send_message3(work,work1)

    #ドット絵
    elif event.message.text == "4":
        print("通過: {}".format(event.message.text))
        with open(path_w1) as f:
            work = f.read()
        with open(path_w2) as f2:
            work1 = f2.read()
        handle_send_message4(work,work1)




def text_save_id(work):
    s = work
    print("取得イヴェントメッセージIDDDDDDDDDDDDDDDD_text_saveID:{}".format(work))
    with open(path_w1, mode='w') as f:
        f.write(s)

def text_save_reply(work):
    s = work
    print("取得イヴェントメッセージIDDDDDDDDDDDDDDDD_text_saveReply:{}".format(work))
    with open(path_w2, mode='w') as f:
        f.write(s)


def flex(event):
    work = event.message.id
    reply_work = event.reply_token
    print("取得イヴェントメッセージIDDDDDDDDDDDDDDDD:{}".format(work))
    text_save_id(work)
    text_save_reply(reply_work)

    # Json展開
    json_open = open('carousel.json', 'r')
    json_data = json.load(json_open)
 
    messages = FlexSendMessage(alt_text="test", contents=json_data)
    print("フレックスメッセージ中身: {}".format(messages))
    if event.reply_token == "00000000000000000000000000000000":
        return
    if event.reply_token == "ffffffffffffffffffffffffffffffff":
        return
        
    line_bot_api.push_message('U0702a57cd35b16d81966cf38edfecb78', messages=messages)


# def handle_textmessage(event):
#     line_bot_api.reply_message(event.reply_token,
#         [
#             #TextSendMessage(text=event.message.text),
#             TextSendMessage(text="顔、目を検知できませんでした。"),
#             #TextSendMessage(text=event.message.id),
#         ]
#         )

#画像受信後処理
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    print("メッセージID")
    print(event.message.id)
    message_content = line_bot_api.get_message_content(event.message.id)
    if not os.path.exists('static'):
        os.mkdir('static/')
    with open("static/" + event.message.id + ".jpg", "wb") as f:
        f.write(message_content.content)
    
    flex(event)


################################################################
###------------------//画像送信処理//------------------------###

#モザイク送信
def handle_send_message(event,relpy):
    #mozaiku(event)
    result = mosic_change.mosic_image(event)
    reply = str(relpy)
    if result:
        line_bot_api.reply_message(
            reply, ImageSendMessage(
                original_content_url=FQDN + "/static/" + event + "_face.jpg",
                preview_image_url=FQDN + "/static/" + event + "_face.jpg",
            )
            )

    # else:
    #     handle_textmessage(event)

# 線画送信
def handle_send_message2(event,reply):
    plt.set_cmap("gray")
    result = art_change.art_image(event)
    reply = str(reply)
    # if result:
    line_bot_api.reply_message(
        reply, ImageSendMessage(
            original_content_url=FQDN + "/static/" + event + "_face.jpg",
            preview_image_url=FQDN + "/static/" + event + "_face.jpg",
        )
    )
    # else:
    #     handle_textmessage(event)
    
# イラスト送信
def handle_send_message3(event,relpy):
    result = illust_change.illust_image(event)
    reply = str(relpy)
    line_bot_api.reply_message(
        reply, ImageSendMessage(
            original_content_url=FQDN + "/static/" + event + "_face.jpg",
            preview_image_url=FQDN + "/static/" + event + "_face.jpg",
        )
        )

# ドット絵送信
def handle_send_message4(event,relpy):
    result = mosic_change.mosic_image(event)
    reply = str(relpy)
    line_bot_api.reply_message(
        reply, ImageSendMessage(
            original_content_url=FQDN + "/static/" + event + "_face.jpg",
            preview_image_url=FQDN + "/static/" + event + "_face.jpg",
        )
        )


# # 画像のパスを関数化
# def get_image_path(event):

#     image_file = event + ".jpg"
#     save_file = event + "_face.jpg"
#     print("イメージファイル: {} // {}".format(image_file, save_file))

#     # 元画像の保存場所をパスとして保管
#     image_path = "static/" + image_file
#     print("イメージパス: {}".format(image_path))

#     # 加工済みの画像の保存場所をパスとして保管
#     output_path = "static/" + save_file
#     print("アウトプットパス: {}".format(output_path))

#     return image_path, output_path



# ################################################################
# ###-------------------------モザイク--------------------------###

# #囲う処理
# def mosic_image(event):
#     bool = True
#     cascade_path = "haarcascade_frontalface_default.xml"
#     cascade_eye_path = "haarcascade_eye.xml"

#     image_path, output_path = get_image_path(event)

#     # ファイル読み込みo
#     image = cv2.imread(image_path)

#     # グレースケール変換
#     image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # カスケード分類器の特徴量を取得する
#     cascade = cv2.CascadeClassifier(cascade_path)
#     cascade_eye = cv2.CascadeClassifier(cascade_eye_path)
#     # 物体認識（顔認識）の実行
#     # image – CV_8U 型の行列．ここに格納されている画像中から物体が検出されます
#     # objects – 矩形を要素とするベクトル．それぞれの矩形は，検出した物体を含みます
#     # scaleFactor – 各画像スケールにおける縮小量を表します
#     # minNeighbors – 物体候補となる矩形は，最低でもこの数だけの近傍矩形を含む必要があります
#     # flags – このパラメータは，新しいカスケードでは利用されません．古いカスケードに対しては，cvHaarDetectObjects 関数の場合と同じ意味を持ちます
#     # minSize – 物体が取り得る最小サイズ．これよりも小さい物体は無視されます
#     facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
#     eyerect = cascade_eye.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(20, 20))
#     print("レクト:{} // {}".format(facerect, eyerect))
#     print("ぎっとぽっど")

#     color = (255, 0, 0)  # 青

#     ratio = 0.05  # 縮小処理時の縮小率(小さいほどモザイクが大きくなる)
#     if len(eyerect) > 0:
#         # for rect in eyerect:
#         #     cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), color, thickness=1)
#         #     print(eyerect)
#         #     print(rect)
           

#         for x, y, w, h in eyerect:  # 引数でeyesで取得した数分forループ
#            # y:はHEIGHT、x:はWEIGHT  fxはxの縮小率、fyはyの縮小率
#            small = cv2.resize(image[y: y + h, x: x + w], None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
#            image[y: y + h, x: x + w] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
           
#     else:
#         bool = False

#     if bool:
#         # 認識結果の保存
#         cv2.imwrite(output_path, image)
#         #cv2.imwrite(output_path2, image)
#         return True
#     else:
#         return False

# ################################################################
# ###-------------------------線画処理--------------------------###

# def art_image(event):

#     image_path, output_path = get_image_path(event)

#     # カーネルを定義
#     kernel = np.ones((5,5), np.uint8)
#     kernel[0,0] = kernel[0,4] = kernel[4,0] = kernel[4,4] = 0
#     # グレースケールで画像を読み込む.
#     # gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     # ImageMagickの convert -flatten x.png y.png に対応
#     img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     if img.shape[2] == 4:
#         mask = img[:,:,3] == 0
#         img[mask] = [255] * 4
#         img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # ノイズ除去が必要なら
#     # gray = cv2.fastNlMeansDenoising(gray, h=20)
#     # 白い部分を膨張させる
#     dilated = cv2.dilate(gray, kernel, iterations=1)
#     # 差をとる
#     diff = cv2.absdiff(dilated, gray)
#     # 白黒反転して2値化
#     # _, contour = cv2.threshold(255 - diff, 240, 255, cv2.THRESH_BINARY)
#     # あるいは
#     image = cv2.adaptiveThreshold(255 - diff, 255,
#                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                     cv2.THRESH_BINARY, 7, 8)

#     cv2.imwrite(output_path, image)

# ################################################################
# ###-------------------------イラスト--------------------------###

# def illust_filter(img, K=20):

#     # グレースケール変換
#     gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

#     # ぼかしでノイズ低減
#     edge = cv2.blur(gray, (3, 3))

#     # Cannyアルゴリズムで輪郭抽出
#     edge = cv2.Canny(edge, 50, 150, apertureSize=3)

#     # 輪郭画像をRGB色空間に変換
#     edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

#     # 画像の減色処理
#     img = np.array(img/K, dtype=np.uint8)
#     img = np.array(img*K, dtype=np.uint8)

#     # 差分を返す
#     return cv2.subtract(img, edge)


# def illust_image(event):
    
#     image_path, output_path = get_image_path(event)

#     # 元画像の読み込み
#     img = cv2.imread(image_path)

#     # 画像のアニメ絵化
#     image = illust_filter(img, 30)

#     # 結果出力
#     cv2.imwrite(output_path, image)

# ################################################################
# ###-------------------------ドット絵--------------------------###

# # 減色処理
# def sub_color(src, K):
#     # 次元数を1落とす
#     Z = src.reshape((-1,3))

#     # float32型に変換
#     Z = np.float32(Z)

#     # 基準の定義
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#     # K-means法で減色
#     ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

#     # UINT8に変換
#     center = np.uint8(center)

#     res = center[label.flatten()]

#     # 配列の次元数と入力画像と同じに戻す
#     return res.reshape((src.shape))

# # モザイク処理
# def mosaic(img, alpha):
#     # 画像の高さ、幅、チャンネル数
#     h, w, ch = img.shape

#     # 縮小→拡大でモザイク加工
#     img = cv2.resize(img,(int(w*alpha), int(h*alpha)))
#     img = cv2.resize(img,(w, h), interpolation=cv2.INTER_NEAREST)

#     return img

# # ドット絵化
# def pixel_art(img, alpha=10, K=4):
#     # モザイク処理
#     img = mosaic(img, alpha)

#     # 減色処理
#     return sub_color(img, K)

# def dot_image(event):
    
#     image_path, output_path = get_image_path(event)

#     # 元画像の読み込み
#     img = cv2.imread(image_path)
 
#     # ドット絵化
#     dst = pixel_art(img, 0.5, 4)
    
#     # 結果を出力
#     print("出力確認:{}".format(output_path))
#     cv2.imwrite(output_path, dst)
# ################################################################

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)