FROM fschlatt/natural-language-processing-exercises:0.0.1

RUN apt-get update 
RUN pip3 install aaransia

ADD language_identification_alphabets.py /code/language_identification_alphabets.py

ENTRYPOINT [ "python3", "/code/language_identification_alphabets.py" ]
