import random, os

from PIL import Image
import mysql
from flask import Flask, request, render_template, Response, jsonify
import os
import mysql.connector
from flask_restful import Api

app = Flask(__name__)
api = Api(app)
app.secret_key = "secret key"
mylist = []
mylist1 = []
mylist2 = ()


@app.route("/", methods=["POST", "GET"])
def display():
    ret = ""
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="sharco"

    )

    mycursor = mydb.cursor()
    sql = "SELECT * FROM images ORDER BY RAND() LIMIT 2"
    mycursor.execute(sql)
    data = mycursor.fetchall()
    mydb.commit()

    for row in data:
        imageObj = []
        imageObj.append('static/img/' + row[1] + '.jpg')
        imageObj.append('show/' + row[1])
        # mylist.append('img/' + row[1] + '.jpg')
        mylist.append(imageObj)
    # print(mylist)

    return render_template("index.html", data=mylist)


@app.route('/show/<string:id>', methods=['GET'])
def get_blog_post(id):
    global mylist2
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="sharco"

    )

    mycursor = mydb.cursor()
    sql = "SELECT * FROM images WHERE imageName = '" + id + "'"
    mycursor.execute(sql)
    data = mycursor.fetchall()
    print(data)
    mydb.commit()

    for row in data:
        mylist2 = row[2]

    sql = "SELECT * FROM images WHERE FIND_IN_SET(Detected,'" + mylist2 + "')>0 ORDER BY RAND() LIMIT 2"
    mycursor.execute(sql)
    data2 = mycursor.fetchall()
    mydb.commit()
    for row1 in data2:
        imageObj1 = []
        imageObj1.append('static/img/' + row1[1] + '.jpg')
        mylist1.append(imageObj1)

    return render_template("show.html", id=id, data=mylist1)


app.debug = True
app.run()
