#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:51:29 2021

Petra lernt programmieren!

Petra wird die nachfolgend aufgelisteten Erfolge feiern:
    
    1. Ein Programm schreiben, dass "Hello World" ausgibt.
    2. Ein Programm schreiben, dass alle Zahlen von 1 bis 100 addiert
    3. Ein Programm schreiben, dass die Sinusfunktion im Intervall [0,1] auswertet
    4. Ein Prgoramm schreiben, dass die Sinusfunktion plottet


@author: jemil
"""


"""
    1. Ein Programm schreiben, dass "Hello World" ausgibt.
"""

# i) Die print-Funktion nimmt als Input einen String (=Buchstabenzeichen abgegrenzt
# durch " ") und zeigt ihn in der Kommandozeile an. Eine Funktion f mit Input x wird aufgerufen,
# durch den befehl f(x).

# ii) Verwende die Hilfefunktion help(), um herauszufinden, wie die Funktion print() Funktioniert.
# Gebe dazu help(print) in die Kommandozeile ein.

# iii) Schreibe in dieses Skript einen Befehl, der "Hallo, ich bin Petra" in der
# Kommandozeile anzeigt.


"""
    2. Ein Programm schreiben, dass alle Zahlen von 1 bis 100 addiert
"""

# i) Bereiten wir uns erst einmal vor. Berechne die Summe der zahlen 1 bis 5, indem
# Du dem Computer sagst, er soll sie addieren. Nennen diese Summe x und drucke sie aus.
# Du kannst dies tun, indem du dem computer mitteilst, dass x = 1+2+3+4+5.
# Anschliessend verwendest Du den print() Befehl, um x auszudrucken


# ii) Alle Zahlen von 1 bis 100 aufschreiben, waere sehr muehsam. Stattdessen
# benutzen wir etwas, das heisst "Schleife" oder auf Englisch "loop".
# Das funktioniert wie folgt. Wir definieren eine Zahl k und sagen dem Computer, er soll 
# diese Zahl k von 1 bis 100 laufen lassen. Der Befehl sieht so aus:
#    for k in range(101):
#       #Mache etwas 
#
# Wie kannst Du dem computer sagen, dass er die Zahlen k alle aufsummieren soll?
# Du koenntest versuchen, wieder eine Zahl x zu definieren und bei jedem durchlauf
# die Zahl k auf die Zahl x zu addieren.

import numpy as np

    


# i) Die Variablen t_1, t_2 definieren
# Definiere diese Variablen, wie Du auch schon unter 3 Variablen definiert hast.
# Lasse Sie laufen von -4 bis 4 und zwar in insgesamt 100 Schritten. 


# ii) Ein Raster generieren, in dem die Werte fuer die Funktion z eingetragen werden
# Verwende den Befehl np.zeros(), um ein Raster der Groesse 100 x 100 zu erstellen.
# Es enthaelt ueberall nur Nullen Nenne dieses Raster "z"  Schaue dir 
# "Variable explorer" auf der rechten Seite an, ob alles gut gelaufen ist.

"""
    3. Ein Programm schreiben, dass die Sinusfunktion im Intervall [0,10] auswertet
"""

# i) Dazu brauchen wir numerische Funktionen, das Modul importiere ich: Du musst dies
# nicht tun.

import numpy as np

# Ab jetzt kann auf Funktionen des Paketes "numpy" fuer numerische Mathematik 
# zugegriffen werden.

# ii) Definiere jetzt eine Folge von 100 Zahlen zwischen 0 und 10 in gleichem Abstand.
# Dazu stellt numpy die Funktion linspace bereit. Erfahre mehr ueber diese Funktion,
# indem Du help(np.linspace) eingibst. Wenn Du die Funktion versteht, benutze sie, um
# die Zahlenfolge zu generieren und sie der Variable t zuzuweisen.

# iii) Wende die Funktion sin() auf t an, um eine Reihe von Sinuswerten zu generieren.
# Nenne die so generierten Werte y.


"""
    4. Ein Prgoramm schreiben, dass die Sinusfunktion plottet
"""

# i) Dazu brauchen wir Funktionen, die grafische Darstellungen ermoeglichen.
# das Modul importiere ich: Du musst dies nicht tun.

import matplotlib.pyplot as plt

# ii) Nutze die Funktion plt.plot(), um y in Abhaengigkeit von t zeichnen zu lassen.


"""
     5. Ein Programm schreiben, dass die werte von sin(t_1*t_2) zeichnet
"""

# Ziel ist es, ein zweidimensionales Bild zu erzeugen von der Funktion z=sin(t_1*t_2).
# Dieses Bild gibt an jeder Stelle an, welchen Wert die Funktion z dort hat. 
# Die Funktion haengt von zwei Variablen ab: t_1 und t_2
#
# Es gibt hier mehrere Schritte, die hintereinander ausgefuehrt werden muessen:
# i) Die Variablen t_1, t_2 definieren
# ii) Ein Raster generieren, in dem die Werte fuer die Funktion z eingetragen werden
# iii) Das Raaster mit Werten fuer die Funktion z fuellen
# iv) Das gefuellte Raster zeichnen lassen



# i) Die Variablen t_1, t_2 definieren
# Definiere diese Variablen, wie Du auch schon unter 3 Variablen definiert hast.
# Lasse Sie laufen von -4 bis 4 und zwar in insgesamt 100 Schritten. 


# ii) Ein Raster generieren, in dem die Werte fuer die Funktion z eingetragen werden
# Verwende den Befehl np.zeros(), um ein Raster der Groesse 100 x 100 zu erstellen.
# Es enthaelt ueberall nur Nullen Nenne dieses Raster "z"  Schaue dir 
# "Variable explorer" auf der rechten Seite an, ob alles gut gelaufen ist.


# iii) Das Raaster mit Werten fuer die Funktion z fuellen
# Verwende nun zwei ineinander verschachtelte Schleifen mit den Laufvariablen k und l
# (sie laufen bis 100) um die Werte von z aufzufuellen. Folgende Bausteine
#  muessen dabei verbunden werden:
#
# Verschachtelte Schleife:
#
# for k in range(100)    
#   for l in range(100)
#       Mache etwas
#
# Zuweisung von Werten:
#
# z=np.sin(t_1[k]*t_2[l])



# iv) Das gefuellte Raster zeichnen lassen
# Nutze die Funktion plt.imshow, um das mit Werten gefuellte Raster zu zeichnen.





























"""
    Loesungen
"""

"""
print("Hello World")

x=0
for k in range(101):
   x=x+k

print(x)

import numpy as np

t=np.linspace(0,10,100)
y=np.sin(t)
print(y)


import matplotlib.pyplot as plt
plt.plot(t,y)


t_1=np.linspace(-4,4,100)
t_2=np.linspace(-4,4,100)

z=np.zeros([100,100])
for k in range(100):
    for l in range(100):
        z[k,l]=np.sin(t_1[k]*t_2[l])
        
plt.imshow(z)

"""