їс
и¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.0.02v2.0.0-rc2-26-g64c3d388ч╚

~
conv2d/kernelVarHandleOp*
shared_nameconv2d/kernel*
dtype0*
_output_shapes
: *
shape: 
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*
dtype0*&
_output_shapes
: 
n
conv2d/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
dtype0*
_output_shapes
: 
В
conv2d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
: @
r
conv2d_1/biasVarHandleOp*
shape:@*
shared_nameconv2d_1/bias*
dtype0*
_output_shapes
: 
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:@
В
conv2d_2/kernelVarHandleOp* 
shared_nameconv2d_2/kernel*
dtype0*
_output_shapes
: *
shape:@@
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*
dtype0*&
_output_shapes
:@@
r
conv2d_2/biasVarHandleOp*
shared_nameconv2d_2/bias*
dtype0*
_output_shapes
: *
shape:@
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
dtype0*
_output_shapes
:@
v
dense/kernelVarHandleOp*
shape:
А╥*
shared_namedense/kernel*
dtype0*
_output_shapes
: 
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0* 
_output_shapes
:
А╥
l

dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
y
dense_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	А
q
dense_1/biasVarHandleOp*
shape:А*
shared_namedense_1/bias*
dtype0*
_output_shapes
: 
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes	
:А
y
dense_2/kernelVarHandleOp*
shape:	А*
shared_namedense_2/kernel*
dtype0*
_output_shapes
: 
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
dtype0*
_output_shapes
:	А
p
dense_2/biasVarHandleOp*
shared_namedense_2/bias*
dtype0*
_output_shapes
: *
shape:
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
shape: *
shared_name	Adam/iter*
dtype0	*
_output_shapes
: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
shape: *
shared_nameAdam/beta_2*
dtype0*
_output_shapes
: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shared_name
Adam/decay*
dtype0*
_output_shapes
: *
shape: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shared_namecount*
dtype0*
_output_shapes
: *
shape: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
М
Adam/conv2d/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *%
shared_nameAdam/conv2d/kernel/m
Е
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*
dtype0*&
_output_shapes
: 
|
Adam/conv2d/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape: *#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
dtype0*
_output_shapes
: 
Р
Adam/conv2d_1/kernel/mVarHandleOp*'
shared_nameAdam/conv2d_1/kernel/m*
dtype0*
_output_shapes
: *
shape: @
Й
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*
dtype0*&
_output_shapes
: @
А
Adam/conv2d_1/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
dtype0*
_output_shapes
:@
Р
Adam/conv2d_2/kernel/mVarHandleOp*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/m*
dtype0*
_output_shapes
: 
Й
*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*
dtype0*&
_output_shapes
:@@
А
Adam/conv2d_2/bias/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
dtype0*
_output_shapes
:@
Д
Adam/dense/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:
А╥*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
dtype0* 
_output_shapes
:
А╥
z
Adam/dense/bias/mVarHandleOp*"
shared_nameAdam/dense/bias/m*
dtype0*
_output_shapes
: *
shape:
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
dtype0*
_output_shapes
:
З
Adam/dense_1/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А*&
shared_nameAdam/dense_1/kernel/m
А
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
dtype0*
_output_shapes
:	А

Adam/dense_1/bias/mVarHandleOp*$
shared_nameAdam/dense_1/bias/m*
dtype0*
_output_shapes
: *
shape:А
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
dtype0*
_output_shapes	
:А
З
Adam/dense_2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А*&
shared_nameAdam/dense_2/kernel/m
А
)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
dtype0*
_output_shapes
:	А
~
Adam/dense_2/bias/mVarHandleOp*$
shared_nameAdam/dense_2/bias/m*
dtype0*
_output_shapes
: *
shape:
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
dtype0*
_output_shapes
:
М
Adam/conv2d/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *%
shared_nameAdam/conv2d/kernel/v
Е
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*
dtype0*&
_output_shapes
: 
|
Adam/conv2d/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: *#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
dtype0*
_output_shapes
: 
Р
Adam/conv2d_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape: @*'
shared_nameAdam/conv2d_1/kernel/v
Й
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*
dtype0*&
_output_shapes
: @
А
Adam/conv2d_1/bias/vVarHandleOp*
shape:@*%
shared_nameAdam/conv2d_1/bias/v*
dtype0*
_output_shapes
: 
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
dtype0*
_output_shapes
:@
Р
Adam/conv2d_2/kernel/vVarHandleOp*'
shared_nameAdam/conv2d_2/kernel/v*
dtype0*
_output_shapes
: *
shape:@@
Й
*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*
dtype0*&
_output_shapes
:@@
А
Adam/conv2d_2/bias/vVarHandleOp*
shape:@*%
shared_nameAdam/conv2d_2/bias/v*
dtype0*
_output_shapes
: 
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
dtype0*
_output_shapes
:@
Д
Adam/dense/kernel/vVarHandleOp*
shape:
А╥*$
shared_nameAdam/dense/kernel/v*
dtype0*
_output_shapes
: 
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
dtype0* 
_output_shapes
:
А╥
z
Adam/dense/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
dtype0*
_output_shapes
:
З
Adam/dense_1/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А*&
shared_nameAdam/dense_1/kernel/v
А
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
dtype0*
_output_shapes
:	А

Adam/dense_1/bias/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:А*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
dtype0*
_output_shapes	
:А
З
Adam/dense_2/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape:	А*&
shared_nameAdam/dense_2/kernel/v
А
)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
dtype0*
_output_shapes
:	А
~
Adam/dense_2/bias/vVarHandleOp*$
shared_nameAdam/dense_2/bias/v*
dtype0*
_output_shapes
: *
shape:
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
ЦV
ConstConst"/device:CPU:0*╤U
value╟UB─U B╜U
ё
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
R
"regularization_losses
#	variables
$trainable_variables
%	keras_api
R
&regularization_losses
'	variables
(trainable_variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
R
0regularization_losses
1	variables
2trainable_variables
3	keras_api
R
4regularization_losses
5	variables
6trainable_variables
7	keras_api
h

8kernel
9bias
:regularization_losses
;	variables
<trainable_variables
=	keras_api
R
>regularization_losses
?	variables
@trainable_variables
A	keras_api
R
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
R
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
R
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
h

Tkernel
Ubias
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
R
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
h

^kernel
_bias
`regularization_losses
a	variables
btrainable_variables
c	keras_api
R
dregularization_losses
e	variables
ftrainable_variables
g	keras_api
░
hiter

ibeta_1

jbeta_2
	kdecay
llearning_ratem┴m┬*m├+m─8m┼9m╞Jm╟Km╚Tm╔Um╩^m╦_m╠v═v╬*v╧+v╨8v╤9v╥Jv╙Kv╘Tv╒Uv╓^v╫_v╪
 
V
0
1
*2
+3
84
95
J6
K7
T8
U9
^10
_11
V
0
1
*2
+3
84
95
J6
K7
T8
U9
^10
_11
Ъ
regularization_losses
mlayer_regularization_losses
nmetrics
onon_trainable_variables
	variables
trainable_variables

players
 
 
 
 
Ъ
regularization_losses
qlayer_regularization_losses
rmetrics
snon_trainable_variables
	variables
trainable_variables

tlayers
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Ъ
regularization_losses
ulayer_regularization_losses
vmetrics
wnon_trainable_variables
	variables
 trainable_variables

xlayers
 
 
 
Ъ
"regularization_losses
ylayer_regularization_losses
zmetrics
{non_trainable_variables
#	variables
$trainable_variables

|layers
 
 
 
Ы
&regularization_losses
}layer_regularization_losses
~metrics
non_trainable_variables
'	variables
(trainable_variables
Аlayers
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
Ю
,regularization_losses
 Бlayer_regularization_losses
Вmetrics
Гnon_trainable_variables
-	variables
.trainable_variables
Дlayers
 
 
 
Ю
0regularization_losses
 Еlayer_regularization_losses
Жmetrics
Зnon_trainable_variables
1	variables
2trainable_variables
Иlayers
 
 
 
Ю
4regularization_losses
 Йlayer_regularization_losses
Кmetrics
Лnon_trainable_variables
5	variables
6trainable_variables
Мlayers
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

80
91
Ю
:regularization_losses
 Нlayer_regularization_losses
Оmetrics
Пnon_trainable_variables
;	variables
<trainable_variables
Рlayers
 
 
 
Ю
>regularization_losses
 Сlayer_regularization_losses
Тmetrics
Уnon_trainable_variables
?	variables
@trainable_variables
Фlayers
 
 
 
Ю
Bregularization_losses
 Хlayer_regularization_losses
Цmetrics
Чnon_trainable_variables
C	variables
Dtrainable_variables
Шlayers
 
 
 
Ю
Fregularization_losses
 Щlayer_regularization_losses
Ъmetrics
Ыnon_trainable_variables
G	variables
Htrainable_variables
Ьlayers
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

J0
K1

J0
K1
Ю
Lregularization_losses
 Эlayer_regularization_losses
Юmetrics
Яnon_trainable_variables
M	variables
Ntrainable_variables
аlayers
 
 
 
Ю
Pregularization_losses
 бlayer_regularization_losses
вmetrics
гnon_trainable_variables
Q	variables
Rtrainable_variables
дlayers
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

T0
U1
Ю
Vregularization_losses
 еlayer_regularization_losses
жmetrics
зnon_trainable_variables
W	variables
Xtrainable_variables
иlayers
 
 
 
Ю
Zregularization_losses
 йlayer_regularization_losses
кmetrics
лnon_trainable_variables
[	variables
\trainable_variables
мlayers
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

^0
_1

^0
_1
Ю
`regularization_losses
 нlayer_regularization_losses
оmetrics
пnon_trainable_variables
a	variables
btrainable_variables
░layers
 
 
 
Ю
dregularization_losses
 ▒layer_regularization_losses
▓metrics
│non_trainable_variables
e	variables
ftrainable_variables
┤layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

╡0
 
v
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 


╢total

╖count
╕
_fn_kwargs
╣regularization_losses
║	variables
╗trainable_variables
╝	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

╢0
╖1
 
б
╣regularization_losses
 ╜layer_regularization_losses
╛metrics
┐non_trainable_variables
║	variables
╗trainable_variables
└layers
 
 

╢0
╖1
 
|z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
П
serving_default_conv2d_inputPlaceholder*$
shape:         22*
dtype0*/
_output_shapes
:         22
┌
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *+
_gradient_op_typePartitionedCall-7045*+
f&R$
"__inference_signature_wrapper_6545*
Tout
2
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
Ъ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOpConst*+
_gradient_op_typePartitionedCall-7110*&
f!R
__inference__traced_save_7109*
Tout
2**
config_proto

CPU

GPU 2J 8*
_output_shapes
: *8
Tin1
/2-	
╣
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/v*)
f$R"
 __inference__traced_restore_7251*
Tout
2**
config_proto

CPU

GPU 2J 8*
_output_shapes
: *7
Tin0
.2,*+
_gradient_op_typePartitionedCall-7252АГ	
я
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_6803

inputs
identityИQ
dropout/rateConst*
valueB
 *
╫#<*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  А?Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:         **@М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: к
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         **@Ь
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         **@R
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: С
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         **@i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         **@w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:         **@q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*/
_output_shapes
:         **@*
T0a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         **@"
identityIdentity:output:0*.
_input_shapes
:         **@:& "
 
_user_specified_nameinputs
М
Ы
)__inference_sequential_layer_call_fn_6522
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityИвStatefulPartitionedCall╜
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*'
_output_shapes
:         *
Tin
2*+
_gradient_op_typePartitionedCall-6507*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6506*
Tout
2**
config_proto

CPU

GPU 2J 8В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : :	 :
 : : 
№
b
F__inference_activation_5_layer_call_and_return_conditional_losses_6371

inputs
identityL
SoftmaxSoftmaxinputs*
T0*'
_output_shapes
:         Y
IdentityIdentitySoftmax:softmax:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
мQ
╞
__inference__traced_save_7109
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_1a978db1f64c413092c737b2da530a93/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Щ
SaveV2/tensor_namesConst"/device:CPU:0*┬
value╕B╡+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:+├
SaveV2/shape_and_slicesConst"/device:CPU:0*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:+ц
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *9
dtypes/
-2+	h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:Ц
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*Ю
_input_shapesМ
Й: : : : @:@:@@:@:
А╥::	А:А:	А:: : : : : : : : : : @:@:@@:@:
А╥::	А:А:	А:: : : @:@:@@:@:
А╥::	А:А:	А:: 2
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2_1SaveV2_1:, :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ 
╦
E
)__inference_activation_layer_call_fn_6738

inputs
identityа
PartitionedCallPartitionedCallinputs*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_6043*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         .. *
Tin
2*+
_gradient_op_typePartitionedCall-6049h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         .. "
identityIdentity:output:0*.
_input_shapes
:         .. :& "
 
_user_specified_nameinputs
лг
Ф
 __inference__traced_restore_7251
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias&
"assignvariableop_2_conv2d_1_kernel$
 assignvariableop_3_conv2d_1_bias&
"assignvariableop_4_conv2d_2_kernel$
 assignvariableop_5_conv2d_2_bias#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias%
!assignvariableop_8_dense_1_kernel#
assignvariableop_9_dense_1_bias&
"assignvariableop_10_dense_2_kernel$
 assignvariableop_11_dense_2_bias!
assignvariableop_12_adam_iter#
assignvariableop_13_adam_beta_1#
assignvariableop_14_adam_beta_2"
assignvariableop_15_adam_decay*
&assignvariableop_16_adam_learning_rate
assignvariableop_17_total
assignvariableop_18_count,
(assignvariableop_19_adam_conv2d_kernel_m*
&assignvariableop_20_adam_conv2d_bias_m.
*assignvariableop_21_adam_conv2d_1_kernel_m,
(assignvariableop_22_adam_conv2d_1_bias_m.
*assignvariableop_23_adam_conv2d_2_kernel_m,
(assignvariableop_24_adam_conv2d_2_bias_m+
'assignvariableop_25_adam_dense_kernel_m)
%assignvariableop_26_adam_dense_bias_m-
)assignvariableop_27_adam_dense_1_kernel_m+
'assignvariableop_28_adam_dense_1_bias_m-
)assignvariableop_29_adam_dense_2_kernel_m+
'assignvariableop_30_adam_dense_2_bias_m,
(assignvariableop_31_adam_conv2d_kernel_v*
&assignvariableop_32_adam_conv2d_bias_v.
*assignvariableop_33_adam_conv2d_1_kernel_v,
(assignvariableop_34_adam_conv2d_1_bias_v.
*assignvariableop_35_adam_conv2d_2_kernel_v,
(assignvariableop_36_adam_conv2d_2_bias_v+
'assignvariableop_37_adam_dense_kernel_v)
%assignvariableop_38_adam_dense_bias_v-
)assignvariableop_39_adam_dense_1_kernel_v+
'assignvariableop_40_adam_dense_1_bias_v-
)assignvariableop_41_adam_dense_2_kernel_v+
'assignvariableop_42_adam_dense_2_bias_v
identity_44ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1Ь
RestoreV2/tensor_namesConst"/device:CPU:0*┬
value╕B╡+B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:+╞
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:+*i
value`B^+B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B °
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*┬
_output_shapesп
м:::::::::::::::::::::::::::::::::::::::::::*9
dtypes/
-2+	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:z
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0~
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:В
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:А
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:В
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:А
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:}
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:Б
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0Д
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:В
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0	*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_iterIdentity_12:output:0*
dtype0	*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:Б
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_1Identity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:Б
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_beta_2Identity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:А
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_decayIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
_output_shapes
:*
T0И
AssignVariableOp_16AssignVariableOp&assignvariableop_16_adam_learning_rateIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:{
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0{
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:К
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv2d_kernel_mIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:И
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_conv2d_bias_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0М
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_1_kernel_mIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:К
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_1_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:М
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_2_kernel_mIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:К
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_2_bias_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
_output_shapes
:*
T0Й
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:З
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:Л
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_1_kernel_mIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
_output_shapes
:*
T0Й
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_1_bias_mIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0Л
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_2_kernel_mIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
_output_shapes
:*
T0Й
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_2_bias_mIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:К
AssignVariableOp_31AssignVariableOp(assignvariableop_31_adam_conv2d_kernel_vIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
_output_shapes
:*
T0И
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_conv2d_bias_vIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:М
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv2d_1_kernel_vIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:К
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_1_bias_vIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:М
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_conv2d_2_kernel_vIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:К
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_conv2d_2_bias_vIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:Й
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_kernel_vIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:З
AssignVariableOp_38AssignVariableOp%assignvariableop_38_adam_dense_bias_vIdentity_38:output:0*
dtype0*
_output_shapes
 P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:Л
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_1_kernel_vIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
_output_shapes
:*
T0Й
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_1_bias_vIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:Л
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_2_kernel_vIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:Й
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_2_bias_vIdentity_42:output:0*
dtype0*
_output_shapes
 М
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:╡
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 Б
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: О
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_44Identity_44:output:0*├
_input_shapes▒
о: :::::::::::::::::::::::::::::::::::::::::::2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_39: : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : 
Щ
_
A__inference_dropout_layer_call_and_return_conditional_losses_6083

inputs

identity_1V
IdentityIdentityinputs*/
_output_shapes
:         .. *
T0c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         .. "!

identity_1Identity_1:output:0*.
_input_shapes
:         .. :& "
 
_user_specified_nameinputs
П
b
F__inference_activation_1_layer_call_and_return_conditional_losses_6108

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         **@b
IdentityIdentityRelu:activations:0*/
_output_shapes
:         **@*
T0"
identityIdentity:output:0*.
_input_shapes
:         **@:& "
 
_user_specified_nameinputs
г;
╝
D__inference_sequential_layer_call_and_return_conditional_losses_6506

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallГ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         .. *
Tin
2*+
_gradient_op_typePartitionedCall-5976*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5970╠
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         .. *+
_gradient_op_typePartitionedCall-6049*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_6043*
Tout
2┬
dropout/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-6095*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6083*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         .. е
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5994*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         **@*
Tin
2*+
_gradient_op_typePartitionedCall-6000╥
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         **@*
Tin
2*+
_gradient_op_typePartitionedCall-6114*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_6108╚
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         **@*+
_gradient_op_typePartitionedCall-6160*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_6148*
Tout
2з
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6024*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6018*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         &&@*
Tin
2╥
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*/
_output_shapes
:         &&@*
Tin
2*+
_gradient_op_typePartitionedCall-6179*O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_6173*
Tout
2**
config_proto

CPU

GPU 2J 8╚
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-6225*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_6213*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         &&@*
Tin
2╗
flatten/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*)
_output_shapes
:         А╥*+
_gradient_op_typePartitionedCall-6242*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_6236С
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_6259*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *+
_gradient_op_typePartitionedCall-6265╟
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2*+
_gradient_op_typePartitionedCall-6287*O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_6281*
Tout
2Я
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         А*+
_gradient_op_typePartitionedCall-6310*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6304╩
activation_4/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*(
_output_shapes
:         А*+
_gradient_op_typePartitionedCall-6332*O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_6326*
Tout
2**
config_proto

CPU

GPU 2J 8Ю
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2*+
_gradient_op_typePartitionedCall-6355*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_6349*
Tout
2╔
activation_5/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_6371*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *+
_gradient_op_typePartitionedCall-6377╕
IdentityIdentity%activation_5/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:
 : : :& "
 
_user_specified_nameinputs: : : : : : : : :	 
Хn
√
D__inference_sequential_layer_call_and_return_conditional_losses_6643

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOp╕
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: и
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         .. о
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: Т
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         .. j
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         .. Y
dropout/dropout/rateConst*
valueB
 *
╫#<*
dtype0*
_output_shapes
: b
dropout/dropout/ShapeShapeactivation/Relu:activations:0*
_output_shapes
:*
T0g
"dropout/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    g
"dropout/dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: д
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:         .. д
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0┬
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*/
_output_shapes
:         .. *
T0┤
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         .. Z
dropout/dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: А
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
T0*
_output_shapes
: й
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*
T0*/
_output_shapes
:         .. Р
dropout/dropout/mulMulactivation/Relu:activations:0dropout/dropout/truediv:z:0*
T0*/
_output_shapes
:         .. З
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:         .. *

SrcT0
Й
dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:         .. ╝
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @┐
conv2d_1/Conv2DConv2Ddropout/dropout/mul_1:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:         **@*
T0*
strides
*
paddingVALID▓
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         **@n
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         **@[
dropout_1/dropout/rateConst*
valueB
 *
╫#<*
dtype0*
_output_shapes
: f
dropout_1/dropout/ShapeShapeactivation_1/Relu:activations:0*
T0*
_output_shapes
:i
$dropout_1/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    i
$dropout_1/dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: и
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:         **@к
$dropout_1/dropout/random_uniform/subSub-dropout_1/dropout/random_uniform/max:output:0-dropout_1/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0╚
$dropout_1/dropout/random_uniform/mulMul7dropout_1/dropout/random_uniform/RandomUniform:output:0(dropout_1/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         **@║
 dropout_1/dropout/random_uniformAdd(dropout_1/dropout/random_uniform/mul:z:0-dropout_1/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         **@\
dropout_1/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?А
dropout_1/dropout/subSub dropout_1/dropout/sub/x:output:0dropout_1/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_1/dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Ж
dropout_1/dropout/truedivRealDiv$dropout_1/dropout/truediv/x:output:0dropout_1/dropout/sub:z:0*
T0*
_output_shapes
: п
dropout_1/dropout/GreaterEqualGreaterEqual$dropout_1/dropout/random_uniform:z:0dropout_1/dropout/rate:output:0*
T0*/
_output_shapes
:         **@Ц
dropout_1/dropout/mulMulactivation_1/Relu:activations:0dropout_1/dropout/truediv:z:0*
T0*/
_output_shapes
:         **@Л
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:         **@П
dropout_1/dropout/mul_1Muldropout_1/dropout/mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:         **@╝
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@┴
conv2d_2/Conv2DConv2Ddropout_1/dropout/mul_1:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         &&@▓
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@Ш
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         &&@n
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         &&@[
dropout_2/dropout/rateConst*
valueB
 *
╫#<*
dtype0*
_output_shapes
: f
dropout_2/dropout/ShapeShapeactivation_2/Relu:activations:0*
T0*
_output_shapes
:i
$dropout_2/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_2/dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: и
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
dtype0*/
_output_shapes
:         &&@*
T0к
$dropout_2/dropout/random_uniform/subSub-dropout_2/dropout/random_uniform/max:output:0-dropout_2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ╚
$dropout_2/dropout/random_uniform/mulMul7dropout_2/dropout/random_uniform/RandomUniform:output:0(dropout_2/dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         &&@║
 dropout_2/dropout/random_uniformAdd(dropout_2/dropout/random_uniform/mul:z:0-dropout_2/dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         &&@\
dropout_2/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?А
dropout_2/dropout/subSub dropout_2/dropout/sub/x:output:0dropout_2/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_2/dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Ж
dropout_2/dropout/truedivRealDiv$dropout_2/dropout/truediv/x:output:0dropout_2/dropout/sub:z:0*
T0*
_output_shapes
: п
dropout_2/dropout/GreaterEqualGreaterEqual$dropout_2/dropout/random_uniform:z:0dropout_2/dropout/rate:output:0*
T0*/
_output_shapes
:         &&@Ц
dropout_2/dropout/mulMulactivation_2/Relu:activations:0dropout_2/dropout/truediv:z:0*
T0*/
_output_shapes
:         &&@Л
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:         &&@*

SrcT0
П
dropout_2/dropout/mul_1Muldropout_2/dropout/mul:z:0dropout_2/dropout/Cast:y:0*/
_output_shapes
:         &&@*
T0f
flatten/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"     i Л
flatten/ReshapeReshapedropout_2/dropout/mul_1:z:0flatten/Reshape/shape:output:0*
T0*)
_output_shapes
:         А╥░
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
А╥З
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0м
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0c
activation_3/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         │
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	АУ
dense_1/MatMulMatMulactivation_3/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А▒
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:АП
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0f
activation_4/ReluReludense_1/BiasAdd:output:0*(
_output_shapes
:         А*
T0│
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	АТ
dense_2/MatMulMatMulactivation_4/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ░
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0k
activation_5/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         ъ
IdentityIdentityactivation_5/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp: : : : : : : :	 :
 : : :& "
 
_user_specified_nameinputs: 
╬
е
$__inference_dense_layer_call_fn_6891

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6265*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_6259*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*0
_input_shapes
:         А╥::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
я
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_6141

inputs
identityИQ
dropout/rateConst*
valueB
 *
╫#<*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:         **@М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: к
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         **@Ь
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         **@R
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: С
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*/
_output_shapes
:         **@*
T0i
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:         **@*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:         **@*

SrcT0
q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         **@a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         **@"
identityIdentity:output:0*.
_input_shapes
:         **@:& "
 
_user_specified_nameinputs
я
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_6206

inputs
identityИQ
dropout/rateConst*
valueB
 *
╫#<*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  А?Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*/
_output_shapes
:         &&@*
T0М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: к
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         &&@Ь
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         &&@R
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: С
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         &&@i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         &&@w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:         &&@q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*/
_output_shapes
:         &&@*
T0a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         &&@"
identityIdentity:output:0*.
_input_shapes
:         &&@:& "
 
_user_specified_nameinputs
Ы
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_6148

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         **@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         **@"!

identity_1Identity_1:output:0*.
_input_shapes
:         **@:& "
 
_user_specified_nameinputs
■
┌
A__inference_dense_1_layer_call_and_return_conditional_losses_6304

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpг
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Аj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Аw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АК
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*.
_input_shapes
:         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ў
b
F__inference_activation_3_layer_call_and_return_conditional_losses_6896

inputs
identityF
ReluReluinputs*'
_output_shapes
:         *
T0Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
√
┌
A__inference_dense_2_layer_call_and_return_conditional_losses_6349

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpг
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Аi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*/
_input_shapes
:         А::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
╧
G
+__inference_activation_2_layer_call_fn_6828

inputs
identityв
PartitionedCallPartitionedCallinputs**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         &&@*
Tin
2*+
_gradient_op_typePartitionedCall-6179*O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_6173*
Tout
2h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         &&@"
identityIdentity:output:0*.
_input_shapes
:         &&@:& "
 
_user_specified_nameinputs
Ы
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_6808

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         **@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         **@"!

identity_1Identity_1:output:0*.
_input_shapes
:         **@:& "
 
_user_specified_nameinputs
╡;
┬
D__inference_sequential_layer_call_and_return_conditional_losses_6419
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallЙ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5970*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         .. *
Tin
2*+
_gradient_op_typePartitionedCall-5976╠
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         .. *+
_gradient_op_typePartitionedCall-6049*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_6043*
Tout
2┬
dropout/PartitionedCallPartitionedCall#activation/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-6095*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6083*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         .. е
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*/
_output_shapes
:         **@*
Tin
2*+
_gradient_op_typePartitionedCall-6000*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5994*
Tout
2**
config_proto

CPU

GPU 2J 8╥
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*/
_output_shapes
:         **@*
Tin
2*+
_gradient_op_typePartitionedCall-6114*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_6108*
Tout
2**
config_proto

CPU

GPU 2J 8╚
dropout_1/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_6148*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         **@*
Tin
2*+
_gradient_op_typePartitionedCall-6160з
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6024*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6018*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         &&@╥
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         &&@*+
_gradient_op_typePartitionedCall-6179*O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_6173*
Tout
2╚
dropout_2/PartitionedCallPartitionedCall%activation_2/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         &&@*+
_gradient_op_typePartitionedCall-6225*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_6213╗
flatten/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*+
_gradient_op_typePartitionedCall-6242*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_6236*
Tout
2**
config_proto

CPU

GPU 2J 8*)
_output_shapes
:         А╥*
Tin
2С
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_6259*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *+
_gradient_op_typePartitionedCall-6265╟
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_6281*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2*+
_gradient_op_typePartitionedCall-6287Я
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         А*+
_gradient_op_typePartitionedCall-6310*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6304*
Tout
2╩
activation_4/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-6332*O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_6326*
Tout
2**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:         А*
Tin
2Ю
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6355*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_6349*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         ╔
activation_5/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *+
_gradient_op_typePartitionedCall-6377*O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_6371*
Tout
2╕
IdentityIdentity%activation_5/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : :	 :
 : : 
╣
B
&__inference_flatten_layer_call_fn_6874

inputs
identityЧ
PartitionedCallPartitionedCallinputs*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_6236*
Tout
2**
config_proto

CPU

GPU 2J 8*)
_output_shapes
:         А╥*
Tin
2*+
_gradient_op_typePartitionedCall-6242b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         А╥"
identityIdentity:output:0*.
_input_shapes
:         &&@:& "
 
_user_specified_nameinputs
╧
G
+__inference_activation_1_layer_call_fn_6783

inputs
identityв
PartitionedCallPartitionedCallinputs*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_6108*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         **@*+
_gradient_op_typePartitionedCall-6114h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         **@"
identityIdentity:output:0*.
_input_shapes
:         **@:& "
 
_user_specified_nameinputs
√
╪
?__inference_dense_layer_call_and_return_conditional_losses_6884

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpд
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
А╥i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*0
_input_shapes
:         А╥::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
·
Х
)__inference_sequential_layer_call_fn_6728

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityИвStatefulPartitionedCall╖
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6506*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2*+
_gradient_op_typePartitionedCall-6507В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
Н
`
D__inference_activation_layer_call_and_return_conditional_losses_6733

inputs
identityN
ReluReluinputs*/
_output_shapes
:         .. *
T0b
IdentityIdentityRelu:activations:0*/
_output_shapes
:         .. *
T0"
identityIdentity:output:0*.
_input_shapes
:         .. :& "
 
_user_specified_nameinputs
Щ
_
A__inference_dropout_layer_call_and_return_conditional_losses_6763

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         .. c

Identity_1IdentityIdentity:output:0*/
_output_shapes
:         .. *
T0"!

identity_1Identity_1:output:0*.
_input_shapes
:         .. :& "
 
_user_specified_nameinputs
я
b
C__inference_dropout_2_layer_call_and_return_conditional_losses_6848

inputs
identityИQ
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *
╫#<C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:         &&@М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: к
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         &&@Ь
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         &&@R
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: С
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         &&@i
dropout/mulMulinputsdropout/truediv:z:0*/
_output_shapes
:         &&@*
T0w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:         &&@*

SrcT0
q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:         &&@a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         &&@"
identityIdentity:output:0*.
_input_shapes
:         &&@:& "
 
_user_specified_nameinputs
·
]
A__inference_flatten_layer_call_and_return_conditional_losses_6236

inputs
identity^
Reshape/shapeConst*
valueB"     i *
dtype0*
_output_shapes
:f
ReshapeReshapeinputsReshape/shape:output:0*
T0*)
_output_shapes
:         А╥Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         А╥"
identityIdentity:output:0*.
_input_shapes
:         &&@:& "
 
_user_specified_nameinputs
■
┌
A__inference_dense_1_layer_call_and_return_conditional_losses_6911

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpг
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Аj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Аw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АК
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
р
Ф
"__inference_signature_wrapper_6545
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityИвStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *+
_gradient_op_typePartitionedCall-6530*(
f#R!
__inference__wrapped_model_5957В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:, (
&
_user_specified_nameconv2d_input: : : : : : : : :	 :
 : : 
Т

┘
@__inference_conv2d_layer_call_and_return_conditional_losses_5970

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpк
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: м
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingVALID*A
_output_shapes/
-:+                            *
T0*
strides
а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+                            *
T0г
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*H
_input_shapes7
5:+                           ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
э
`
A__inference_dropout_layer_call_and_return_conditional_losses_6076

inputs
identityИQ
dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *
╫#<C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:         .. М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0к
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         .. Ь
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*/
_output_shapes
:         .. R
dropout/sub/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: С
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         .. i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         .. w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*/
_output_shapes
:         .. *

SrcT0
q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*/
_output_shapes
:         .. *
T0a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         .. "
identityIdentity:output:0*.
_input_shapes
:         .. :& "
 
_user_specified_nameinputs
Ы
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_6213

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         &&@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         &&@"!

identity_1Identity_1:output:0*.
_input_shapes
:         &&@:& "
 
_user_specified_nameinputs
№
b
F__inference_activation_5_layer_call_and_return_conditional_losses_6950

inputs
identityL
SoftmaxSoftmaxinputs*'
_output_shapes
:         *
T0Y
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
╥
з
&__inference_dense_1_layer_call_fn_6918

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         А*+
_gradient_op_typePartitionedCall-6310*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6304*
Tout
2Г
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
·
Х
)__inference_sequential_layer_call_fn_6711

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityИвStatefulPartitionedCall╖
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*+
_gradient_op_typePartitionedCall-6455*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6454*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
ы?
м
D__inference_sequential_layer_call_and_return_conditional_losses_6385
conv2d_input)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallЙ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_input%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-5976*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5970*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         .. *
Tin
2╠
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         .. *+
_gradient_op_typePartitionedCall-6049*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_6043*
Tout
2╥
dropout/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6076*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         .. *
Tin
2*+
_gradient_op_typePartitionedCall-6087н
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6000*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5994*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         **@*
Tin
2╥
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         **@*
Tin
2*+
_gradient_op_typePartitionedCall-6114*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_6108·
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall*+
_gradient_op_typePartitionedCall-6152*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_6141*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         **@п
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6018*
Tout
2**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         &&@*
Tin
2*+
_gradient_op_typePartitionedCall-6024╥
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:         &&@*+
_gradient_op_typePartitionedCall-6179*O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_6173*
Tout
2**
config_proto

CPU

GPU 2J 8№
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*+
_gradient_op_typePartitionedCall-6217*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_6206*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         &&@├
flatten/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_6236*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*)
_output_shapes
:         А╥*+
_gradient_op_typePartitionedCall-6242С
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_6259*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2*+
_gradient_op_typePartitionedCall-6265╟
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_6281*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *+
_gradient_op_typePartitionedCall-6287Я
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6304*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         А*+
_gradient_op_typePartitionedCall-6310╩
activation_4/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:         А*
Tin
2*+
_gradient_op_typePartitionedCall-6332*O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_6326*
Tout
2Ю
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_6349*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *+
_gradient_op_typePartitionedCall-6355╔
activation_5/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-6377*O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_6371*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2в
IdentityIdentity%activation_5/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall: : : : : : : :	 :
 : : :, (
&
_user_specified_nameconv2d_input: 
═
a
(__inference_dropout_1_layer_call_fn_6813

inputs
identityИвStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_6141*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         **@*+
_gradient_op_typePartitionedCall-6152К
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:         **@*
T0"
identityIdentity:output:0*.
_input_shapes
:         **@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
·
]
A__inference_flatten_layer_call_and_return_conditional_losses_6869

inputs
identity^
Reshape/shapeConst*
valueB"     i *
dtype0*
_output_shapes
:f
ReshapeReshapeinputsReshape/shape:output:0*
T0*)
_output_shapes
:         А╥Z
IdentityIdentityReshape:output:0*)
_output_shapes
:         А╥*
T0"
identityIdentity:output:0*.
_input_shapes
:         &&@:& "
 
_user_specified_nameinputs
║
G
+__inference_activation_4_layer_call_fn_6928

inputs
identityЫ
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         А*+
_gradient_op_typePartitionedCall-6332*O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_6326a
IdentityIdentityPartitionedCall:output:0*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╖
G
+__inference_activation_5_layer_call_fn_6955

inputs
identityЪ
PartitionedCallPartitionedCallinputs*
Tin
2*'
_output_shapes
:         *+
_gradient_op_typePartitionedCall-6377*O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_6371*
Tout
2**
config_proto

CPU

GPU 2J 8`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
╔
D
(__inference_dropout_2_layer_call_fn_6863

inputs
identityЯ
PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-6225*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_6213*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         &&@h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:         &&@*
T0"
identityIdentity:output:0*.
_input_shapes
:         &&@:& "
 
_user_specified_nameinputs
·
b
F__inference_activation_4_layer_call_and_return_conditional_losses_6326

inputs
identityG
ReluReluinputs*
T0*(
_output_shapes
:         А[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Ы
a
C__inference_dropout_2_layer_call_and_return_conditional_losses_6853

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:         &&@c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:         &&@"!

identity_1Identity_1:output:0*.
_input_shapes
:         &&@:& "
 
_user_specified_nameinputs
╖
G
+__inference_activation_3_layer_call_fn_6901

inputs
identityЪ
PartitionedCallPartitionedCallinputs*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         *+
_gradient_op_typePartitionedCall-6287*O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_6281`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
═
a
(__inference_dropout_2_layer_call_fn_6858

inputs
identityИвStatefulPartitionedCallп
StatefulPartitionedCallStatefulPartitionedCallinputs*+
_gradient_op_typePartitionedCall-6217*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_6206*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         &&@К
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         &&@"
identityIdentity:output:0*.
_input_shapes
:         &&@22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
√
╪
?__inference_dense_layer_call_and_return_conditional_losses_6259

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpд
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
А╥i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*0
_input_shapes
:         А╥::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ў
b
F__inference_activation_3_layer_call_and_return_conditional_losses_6281

inputs
identityF
ReluReluinputs*
T0*'
_output_shapes
:         Z
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
Н
`
D__inference_activation_layer_call_and_return_conditional_losses_6043

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         .. b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         .. "
identityIdentity:output:0*.
_input_shapes
:         .. :& "
 
_user_specified_nameinputs
Ф

█
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5994

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpк
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @м
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
paddingVALID*A
_output_shapes/
-:+                           @*
T0*
strides
а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+                           @*
T0г
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
√
┌
A__inference_dense_2_layer_call_and_return_conditional_losses_6938

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpг
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Аi
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
╔
_
&__inference_dropout_layer_call_fn_6768

inputs
identityИвStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputs*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6076*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         .. *+
_gradient_op_typePartitionedCall-6087К
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         .. "
identityIdentity:output:0*.
_input_shapes
:         .. 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
М
Ы
)__inference_sequential_layer_call_fn_6470
conv2d_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityИвStatefulPartitionedCall╜
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*+
_gradient_op_typePartitionedCall-6455*M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6454*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :	 :
 : : :, (
&
_user_specified_nameconv2d_input: : : : : : : 
┼
B
&__inference_dropout_layer_call_fn_6773

inputs
identityЭ
PartitionedCallPartitionedCallinputs*+
_gradient_op_typePartitionedCall-6095*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6083*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         .. h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         .. "
identityIdentity:output:0*.
_input_shapes
:         .. :& "
 
_user_specified_nameinputs
а
и
'__inference_conv2d_2_layer_call_fn_6029

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6024*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6018*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+                           @Ь
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
┴I
ф	
__inference__wrapped_model_5957
conv2d_input4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource6
2sequential_conv2d_1_conv2d_readvariableop_resource7
3sequential_conv2d_1_biasadd_readvariableop_resource6
2sequential_conv2d_2_conv2d_readvariableop_resource7
3sequential_conv2d_2_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identityИв(sequential/conv2d/BiasAdd/ReadVariableOpв'sequential/conv2d/Conv2D/ReadVariableOpв*sequential/conv2d_1/BiasAdd/ReadVariableOpв)sequential/conv2d_1/Conv2D/ReadVariableOpв*sequential/conv2d_2/BiasAdd/ReadVariableOpв)sequential/conv2d_2/Conv2D/ReadVariableOpв'sequential/dense/BiasAdd/ReadVariableOpв&sequential/dense/MatMul/ReadVariableOpв)sequential/dense_1/BiasAdd/ReadVariableOpв(sequential/dense_1/MatMul/ReadVariableOpв)sequential/dense_2/BiasAdd/ReadVariableOpв(sequential/dense_2/MatMul/ReadVariableOp╬
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: ─
sequential/conv2d/Conv2DConv2Dconv2d_input/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         .. ─
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: │
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         .. *
T0А
sequential/activation/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         .. Л
sequential/dropout/IdentityIdentity(sequential/activation/Relu:activations:0*
T0*/
_output_shapes
:         .. ╥
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @р
sequential/conv2d_1/Conv2DConv2D$sequential/dropout/Identity:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:         **@*
T0*
strides
*
paddingVALID╚
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@╣
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         **@*
T0Д
sequential/activation_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         **@П
sequential/dropout_1/IdentityIdentity*sequential/activation_1/Relu:activations:0*
T0*/
_output_shapes
:         **@╥
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@т
sequential/conv2d_2/Conv2DConv2D&sequential/dropout_1/Identity:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:         &&@*
T0*
strides
╚
*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@╣
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         &&@Д
sequential/activation_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         &&@П
sequential/dropout_2/IdentityIdentity*sequential/activation_2/Relu:activations:0*
T0*/
_output_shapes
:         &&@q
 sequential/flatten/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"     i м
sequential/flatten/ReshapeReshape&sequential/dropout_2/Identity:output:0)sequential/flatten/Reshape/shape:output:0*
T0*)
_output_shapes
:         А╥╞
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
А╥и
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0┬
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:й
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         y
sequential/activation_3/ReluRelu!sequential/dense/BiasAdd:output:0*'
_output_shapes
:         *
T0╔
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А┤
sequential/dense_1/MatMulMatMul*sequential/activation_3/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╟
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:А░
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0|
sequential/activation_4/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         А╔
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А│
sequential/dense_2/MatMulMatMul*sequential/activation_4/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ╞
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:п
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Б
sequential/activation_5/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         ∙
IdentityIdentity)sequential/activation_5/Softmax:softmax:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp:, (
&
_user_specified_nameconv2d_input: : : : : : : : :	 :
 : : 
П
b
F__inference_activation_2_layer_call_and_return_conditional_losses_6173

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         &&@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         &&@"
identityIdentity:output:0*.
_input_shapes
:         &&@:& "
 
_user_specified_nameinputs
Ь
ж
%__inference_conv2d_layer_call_fn_5981

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-5976*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5970*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*A
_output_shapes/
-:+                            Ь
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
│=
√
D__inference_sequential_layer_call_and_return_conditional_losses_6694

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИвconv2d/BiasAdd/ReadVariableOpвconv2d/Conv2D/ReadVariableOpвconv2d_1/BiasAdd/ReadVariableOpвconv2d_1/Conv2D/ReadVariableOpвconv2d_2/BiasAdd/ReadVariableOpвconv2d_2/Conv2D/ReadVariableOpвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOp╕
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: и
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:         .. *
T0*
strides
*
paddingVALIDо
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: Т
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         .. *
T0j
activation/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:         .. u
dropout/IdentityIdentityactivation/Relu:activations:0*
T0*/
_output_shapes
:         .. ╝
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
: @┐
conv2d_1/Conv2DConv2Ddropout/Identity:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:         **@*
T0*
strides
▓
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         **@*
T0n
activation_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         **@y
dropout_1/IdentityIdentityactivation_1/Relu:activations:0*/
_output_shapes
:         **@*
T0╝
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@┴
conv2d_2/Conv2DConv2Ddropout_1/Identity:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:         &&@*
T0*
strides
▓
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@Ш
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         &&@*
T0n
activation_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:         &&@y
dropout_2/IdentityIdentityactivation_2/Relu:activations:0*
T0*/
_output_shapes
:         &&@f
flatten/Reshape/shapeConst*
valueB"     i *
dtype0*
_output_shapes
:Л
flatten/ReshapeReshapedropout_2/Identity:output:0flatten/Reshape/shape:output:0*)
_output_shapes
:         А╥*
T0░
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
А╥З
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         м
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         c
activation_3/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:         │
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	АУ
dense_1/MatMulMatMulactivation_3/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А▒
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:АП
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аf
activation_4/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:         А│
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	АТ
dense_2/MatMulMatMulactivation_4/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ░
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         k
activation_5/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:         ъ
IdentityIdentityactivation_5/Softmax:softmax:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
э
`
A__inference_dropout_layer_call_and_return_conditional_losses_6758

inputs
identityИQ
dropout/rateConst*
valueB
 *
╫#<*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  А?*
dtype0*
_output_shapes
: Ф
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*/
_output_shapes
:         .. М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: к
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*/
_output_shapes
:         .. Ь
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*/
_output_shapes
:         .. *
T0R
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  А?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: С
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*/
_output_shapes
:         .. i
dropout/mulMulinputsdropout/truediv:z:0*
T0*/
_output_shapes
:         .. w
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*/
_output_shapes
:         .. q
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*/
_output_shapes
:         .. *
T0a
IdentityIdentitydropout/mul_1:z:0*
T0*/
_output_shapes
:         .. "
identityIdentity:output:0*.
_input_shapes
:         .. :& "
 
_user_specified_nameinputs
╤
з
&__inference_dense_2_layer_call_fn_6945

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6355*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_6349*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
┘?
ж
D__inference_sequential_layer_call_and_return_conditional_losses_6454

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_2+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_2+
'conv2d_2_statefulpartitionedcall_args_1+
'conv2d_2_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2*
&dense_2_statefulpartitionedcall_args_1*
&dense_2_statefulpartitionedcall_args_2
identityИвconv2d/StatefulPartitionedCallв conv2d_1/StatefulPartitionedCallв conv2d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdropout/StatefulPartitionedCallв!dropout_1/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallГ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputs%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-5976*I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5970*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         .. ╠
activation/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-6049*M
fHRF
D__inference_activation_layer_call_and_return_conditional_losses_6043*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         .. ╥
dropout/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         .. *+
_gradient_op_typePartitionedCall-6087*J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6076н
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5994*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         **@*+
_gradient_op_typePartitionedCall-6000╥
activation_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*O
fJRH
F__inference_activation_1_layer_call_and_return_conditional_losses_6108*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         **@*+
_gradient_op_typePartitionedCall-6114·
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0 ^dropout/StatefulPartitionedCall**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         **@*+
_gradient_op_typePartitionedCall-6152*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_6141*
Tout
2п
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0'conv2d_2_statefulpartitionedcall_args_1'conv2d_2_statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         &&@*+
_gradient_op_typePartitionedCall-6024*K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6018*
Tout
2╥
activation_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-6179*O
fJRH
F__inference_activation_2_layer_call_and_return_conditional_losses_6173*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         &&@№
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*+
_gradient_op_typePartitionedCall-6217*L
fGRE
C__inference_dropout_2_layer_call_and_return_conditional_losses_6206*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*/
_output_shapes
:         &&@├
flatten/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-6242*J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_6236*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*)
_output_shapes
:         А╥С
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6265*H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_6259*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2╟
activation_3/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-6287*O
fJRH
F__inference_activation_3_layer_call_and_return_conditional_losses_6281*
Tout
2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         *
Tin
2Я
dense_1/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*
Tout
2**
config_proto

CPU

GPU 2J 8*(
_output_shapes
:         А*
Tin
2*+
_gradient_op_typePartitionedCall-6310*J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_6304╩
activation_4/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0**
config_proto

CPU

GPU 2J 8*
Tin
2*(
_output_shapes
:         А*+
_gradient_op_typePartitionedCall-6332*O
fJRH
F__inference_activation_4_layer_call_and_return_conditional_losses_6326*
Tout
2Ю
dense_2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0&dense_2_statefulpartitionedcall_args_1&dense_2_statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6355*J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_6349*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         ╔
activation_5/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*+
_gradient_op_typePartitionedCall-6377*O
fJRH
F__inference_activation_5_layer_call_and_return_conditional_losses_6371*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         в
IdentityIdentity%activation_5/PartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*^
_input_shapesM
K:         22::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : 
П
b
F__inference_activation_2_layer_call_and_return_conditional_losses_6823

inputs
identityN
ReluReluinputs*/
_output_shapes
:         &&@*
T0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         &&@"
identityIdentity:output:0*.
_input_shapes
:         &&@:& "
 
_user_specified_nameinputs
П
b
F__inference_activation_1_layer_call_and_return_conditional_losses_6778

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:         **@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:         **@"
identityIdentity:output:0*.
_input_shapes
:         **@:& "
 
_user_specified_nameinputs
╔
D
(__inference_dropout_1_layer_call_fn_6818

inputs
identityЯ
PartitionedCallPartitionedCallinputs**
config_proto

CPU

GPU 2J 8*/
_output_shapes
:         **@*
Tin
2*+
_gradient_op_typePartitionedCall-6160*L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_6148*
Tout
2h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         **@"
identityIdentity:output:0*.
_input_shapes
:         **@:& "
 
_user_specified_nameinputs
·
b
F__inference_activation_4_layer_call_and_return_conditional_losses_6923

inputs
identityG
ReluReluinputs*(
_output_shapes
:         А*
T0[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
Ф

█
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6018

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвConv2D/ReadVariableOpк
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*&
_output_shapes
:@@м
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+                           @а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:@П
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @г
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
а
и
'__inference_conv2d_1_layer_call_fn_6005

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*+
_gradient_op_typePartitionedCall-6000*K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5994*
Tout
2**
config_proto

CPU

GPU 2J 8*A
_output_shapes/
-:+                           @*
Tin
2Ь
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*H
_input_shapes7
5:+                            ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*┴
serving_defaultн
M
conv2d_input=
serving_default_conv2d_input:0         22@
activation_50
StatefulPartitionedCall:0         tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:шС
чN
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-3
layer-11
layer-12
layer_with_weights-4
layer-13
layer-14
layer_with_weights-5
layer-15
layer-16
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+┘&call_and_return_all_conditional_losses
┌__call__
█_default_save_signature"ЩJ
_tf_keras_sequential·I{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 50, 50, 1], "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 13, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 50, 50, 1], "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 13, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
╜
regularization_losses
	variables
trainable_variables
	keras_api
+▄&call_and_return_all_conditional_losses
▌__call__"м
_tf_keras_layerТ{"class_name": "InputLayer", "name": "conv2d_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 50, 50, 1], "config": {"batch_input_shape": [null, 50, 50, 1], "dtype": "float32", "sparse": false, "name": "conv2d_input"}}
б

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
+▐&call_and_return_all_conditional_losses
▀__call__"·
_tf_keras_layerр{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 50, 50, 1], "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 50, 50, 1], "dtype": "float32", "filters": 32, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
Э
"regularization_losses
#	variables
$trainable_variables
%	keras_api
+р&call_and_return_all_conditional_losses
с__call__"М
_tf_keras_layerЄ{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
о
&regularization_losses
'	variables
(trainable_variables
)	keras_api
+т&call_and_return_all_conditional_losses
у__call__"Э
_tf_keras_layerГ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}
ё

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
+ф&call_and_return_all_conditional_losses
х__call__"╩
_tf_keras_layer░{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
б
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+ц&call_and_return_all_conditional_losses
ч__call__"Р
_tf_keras_layerЎ{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
▓
4regularization_losses
5	variables
6trainable_variables
7	keras_api
+ш&call_and_return_all_conditional_losses
щ__call__"б
_tf_keras_layerЗ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}
ё

8kernel
9bias
:regularization_losses
;	variables
<trainable_variables
=	keras_api
+ъ&call_and_return_all_conditional_losses
ы__call__"╩
_tf_keras_layer░{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
б
>regularization_losses
?	variables
@trainable_variables
A	keras_api
+ь&call_and_return_all_conditional_losses
э__call__"Р
_tf_keras_layerЎ{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
▓
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+ю&call_and_return_all_conditional_losses
я__call__"б
_tf_keras_layerЗ{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.01, "noise_shape": null, "seed": null}}
о
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+Ё&call_and_return_all_conditional_losses
ё__call__"Э
_tf_keras_layerГ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
є

Jkernel
Kbias
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
+Є&call_and_return_all_conditional_losses
є__call__"╠
_tf_keras_layer▓{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 92416}}}}
б
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
+Ї&call_and_return_all_conditional_losses
ї__call__"Р
_tf_keras_layerЎ{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
ї

Tkernel
Ubias
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
+Ў&call_and_return_all_conditional_losses
ў__call__"╬
_tf_keras_layer┤{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}}
б
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
+°&call_and_return_all_conditional_losses
∙__call__"Р
_tf_keras_layerЎ{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
Ў

^kernel
_bias
`regularization_losses
a	variables
btrainable_variables
c	keras_api
+·&call_and_return_all_conditional_losses
√__call__"╧
_tf_keras_layer╡{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 13, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
д
dregularization_losses
e	variables
ftrainable_variables
g	keras_api
+№&call_and_return_all_conditional_losses
¤__call__"У
_tf_keras_layer∙{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "softmax"}}
├
hiter

ibeta_1

jbeta_2
	kdecay
llearning_ratem┴m┬*m├+m─8m┼9m╞Jm╟Km╚Tm╔Um╩^m╦_m╠v═v╬*v╧+v╨8v╤9v╥Jv╙Kv╘Tv╒Uv╓^v╫_v╪"
	optimizer
 "
trackable_list_wrapper
v
0
1
*2
+3
84
95
J6
K7
T8
U9
^10
_11"
trackable_list_wrapper
v
0
1
*2
+3
84
95
J6
K7
T8
U9
^10
_11"
trackable_list_wrapper
╗
regularization_losses
mlayer_regularization_losses
nmetrics
onon_trainable_variables
	variables
trainable_variables

players
┌__call__
█_default_save_signature
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
-
■serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
regularization_losses
qlayer_regularization_losses
rmetrics
snon_trainable_variables
	variables
trainable_variables

tlayers
▌__call__
+▄&call_and_return_all_conditional_losses
'▄"call_and_return_conditional_losses"
_generic_user_object
':% 2conv2d/kernel
: 2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Э
regularization_losses
ulayer_regularization_losses
vmetrics
wnon_trainable_variables
	variables
 trainable_variables

xlayers
▀__call__
+▐&call_and_return_all_conditional_losses
'▐"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
"regularization_losses
ylayer_regularization_losses
zmetrics
{non_trainable_variables
#	variables
$trainable_variables

|layers
с__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
&regularization_losses
}layer_regularization_losses
~metrics
non_trainable_variables
'	variables
(trainable_variables
Аlayers
у__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_1/kernel
:@2conv2d_1/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
б
,regularization_losses
 Бlayer_regularization_losses
Вmetrics
Гnon_trainable_variables
-	variables
.trainable_variables
Дlayers
х__call__
+ф&call_and_return_all_conditional_losses
'ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
0regularization_losses
 Еlayer_regularization_losses
Жmetrics
Зnon_trainable_variables
1	variables
2trainable_variables
Иlayers
ч__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
4regularization_losses
 Йlayer_regularization_losses
Кmetrics
Лnon_trainable_variables
5	variables
6trainable_variables
Мlayers
щ__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
б
:regularization_losses
 Нlayer_regularization_losses
Оmetrics
Пnon_trainable_variables
;	variables
<trainable_variables
Рlayers
ы__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
>regularization_losses
 Сlayer_regularization_losses
Тmetrics
Уnon_trainable_variables
?	variables
@trainable_variables
Фlayers
э__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
Bregularization_losses
 Хlayer_regularization_losses
Цmetrics
Чnon_trainable_variables
C	variables
Dtrainable_variables
Шlayers
я__call__
+ю&call_and_return_all_conditional_losses
'ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
Fregularization_losses
 Щlayer_regularization_losses
Ъmetrics
Ыnon_trainable_variables
G	variables
Htrainable_variables
Ьlayers
ё__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
 :
А╥2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
б
Lregularization_losses
 Эlayer_regularization_losses
Юmetrics
Яnon_trainable_variables
M	variables
Ntrainable_variables
аlayers
є__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
Pregularization_losses
 бlayer_regularization_losses
вmetrics
гnon_trainable_variables
Q	variables
Rtrainable_variables
дlayers
ї__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
!:	А2dense_1/kernel
:А2dense_1/bias
 "
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
б
Vregularization_losses
 еlayer_regularization_losses
жmetrics
зnon_trainable_variables
W	variables
Xtrainable_variables
иlayers
ў__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
Zregularization_losses
 йlayer_regularization_losses
кmetrics
лnon_trainable_variables
[	variables
\trainable_variables
мlayers
∙__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
!:	А2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
б
`regularization_losses
 нlayer_regularization_losses
оmetrics
пnon_trainable_variables
a	variables
btrainable_variables
░layers
√__call__
+·&call_and_return_all_conditional_losses
'·"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
dregularization_losses
 ▒layer_regularization_losses
▓metrics
│non_trainable_variables
e	variables
ftrainable_variables
┤layers
¤__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
(
╡0"
trackable_list_wrapper
 "
trackable_list_wrapper
Ц
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
г

╢total

╖count
╕
_fn_kwargs
╣regularization_losses
║	variables
╗trainable_variables
╝	keras_api
+ &call_and_return_all_conditional_losses
А__call__"х
_tf_keras_layer╦{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
╢0
╖1"
trackable_list_wrapper
 "
trackable_list_wrapper
д
╣regularization_losses
 ╜layer_regularization_losses
╛metrics
┐non_trainable_variables
║	variables
╗trainable_variables
└layers
А__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
╢0
╖1"
trackable_list_wrapper
 "
trackable_list_wrapper
,:* 2Adam/conv2d/kernel/m
: 2Adam/conv2d/bias/m
.:, @2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
.:,@@2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
%:#
А╥2Adam/dense/kernel/m
:2Adam/dense/bias/m
&:$	А2Adam/dense_1/kernel/m
 :А2Adam/dense_1/bias/m
&:$	А2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
,:* 2Adam/conv2d/kernel/v
: 2Adam/conv2d/bias/v
.:, @2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
.:,@@2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
%:#
А╥2Adam/dense/kernel/v
:2Adam/dense/bias/v
&:$	А2Adam/dense_1/kernel/v
 :А2Adam/dense_1/bias/v
&:$	А2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
▐2█
D__inference_sequential_layer_call_and_return_conditional_losses_6643
D__inference_sequential_layer_call_and_return_conditional_losses_6419
D__inference_sequential_layer_call_and_return_conditional_losses_6385
D__inference_sequential_layer_call_and_return_conditional_losses_6694└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Є2я
)__inference_sequential_layer_call_fn_6711
)__inference_sequential_layer_call_fn_6470
)__inference_sequential_layer_call_fn_6522
)__inference_sequential_layer_call_fn_6728└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъ2ч
__inference__wrapped_model_5957├
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *3в0
.К+
conv2d_input         22
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
Я2Ь
@__inference_conv2d_layer_call_and_return_conditional_losses_5970╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
Д2Б
%__inference_conv2d_layer_call_fn_5981╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           
ю2ы
D__inference_activation_layer_call_and_return_conditional_losses_6733в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╙2╨
)__inference_activation_layer_call_fn_6738в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
└2╜
A__inference_dropout_layer_call_and_return_conditional_losses_6763
A__inference_dropout_layer_call_and_return_conditional_losses_6758┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
К2З
&__inference_dropout_layer_call_fn_6768
&__inference_dropout_layer_call_fn_6773┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
б2Ю
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5994╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
Ж2Г
'__inference_conv2d_1_layer_call_fn_6005╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                            
Ё2э
F__inference_activation_1_layer_call_and_return_conditional_losses_6778в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_1_layer_call_fn_6783в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
─2┴
C__inference_dropout_1_layer_call_and_return_conditional_losses_6808
C__inference_dropout_1_layer_call_and_return_conditional_losses_6803┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
О2Л
(__inference_dropout_1_layer_call_fn_6813
(__inference_dropout_1_layer_call_fn_6818┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
б2Ю
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6018╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
Ж2Г
'__inference_conv2d_2_layer_call_fn_6029╫
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
2К/+                           @
Ё2э
F__inference_activation_2_layer_call_and_return_conditional_losses_6823в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_2_layer_call_fn_6828в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
─2┴
C__inference_dropout_2_layer_call_and_return_conditional_losses_6853
C__inference_dropout_2_layer_call_and_return_conditional_losses_6848┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
О2Л
(__inference_dropout_2_layer_call_fn_6863
(__inference_dropout_2_layer_call_fn_6858┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ы2ш
A__inference_flatten_layer_call_and_return_conditional_losses_6869в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_flatten_layer_call_fn_6874в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
щ2ц
?__inference_dense_layer_call_and_return_conditional_losses_6884в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╬2╦
$__inference_dense_layer_call_fn_6891в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_activation_3_layer_call_and_return_conditional_losses_6896в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_3_layer_call_fn_6901в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_dense_1_layer_call_and_return_conditional_losses_6911в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_1_layer_call_fn_6918в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_activation_4_layer_call_and_return_conditional_losses_6923в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_4_layer_call_fn_6928в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_dense_2_layer_call_and_return_conditional_losses_6938в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_2_layer_call_fn_6945в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ё2э
F__inference_activation_5_layer_call_and_return_conditional_losses_6950в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_activation_5_layer_call_fn_6955в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
6B4
"__inference_signature_wrapper_6545conv2d_input
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 ┴
"__inference_signature_wrapper_6545Ъ*+89JKTU^_MвJ
в 
Cк@
>
conv2d_input.К+
conv2d_input         22";к8
6
activation_5&К#
activation_5         в
F__inference_activation_3_layer_call_and_return_conditional_losses_6896X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ К
+__inference_activation_1_layer_call_fn_6783[7в4
-в*
(К%
inputs         **@
к " К         **@z
+__inference_activation_3_layer_call_fn_6901K/в,
%в"
 К
inputs         
к "К         ▓
F__inference_activation_1_layer_call_and_return_conditional_losses_6778h7в4
-в*
(К%
inputs         **@
к "-в*
#К 
0         **@
Ъ Л
(__inference_dropout_1_layer_call_fn_6813_;в8
1в.
(К%
inputs         **@
p
к " К         **@п
'__inference_conv2d_2_layer_call_fn_6029Г89IвF
?в<
:К7
inputs+                           @
к "2К/+                           @Л
(__inference_dropout_1_layer_call_fn_6818_;в8
1в.
(К%
inputs         **@
p 
к " К         **@Ь
)__inference_sequential_layer_call_fn_6470o*+89JKTU^_EвB
;в8
.К+
conv2d_input         22
p

 
к "К         Ь
)__inference_sequential_layer_call_fn_6522o*+89JKTU^_EвB
;в8
.К+
conv2d_input         22
p 

 
к "К         z
+__inference_activation_5_layer_call_fn_6955K/в,
%в"
 К
inputs         
к "К         ─
D__inference_sequential_layer_call_and_return_conditional_losses_6419|*+89JKTU^_EвB
;в8
.К+
conv2d_input         22
p 

 
к "%в"
К
0         
Ъ о
__inference__wrapped_model_5957К*+89JKTU^_=в:
3в0
.К+
conv2d_input         22
к ";к8
6
activation_5&К#
activation_5         │
C__inference_dropout_1_layer_call_and_return_conditional_losses_6803l;в8
1в.
(К%
inputs         **@
p
к "-в*
#К 
0         **@
Ъ z
&__inference_dense_2_layer_call_fn_6945P^_0в-
&в#
!К
inputs         А
к "К         ▒
A__inference_dropout_layer_call_and_return_conditional_losses_6763l;в8
1в.
(К%
inputs         .. 
p 
к "-в*
#К 
0         .. 
Ъ ▒
A__inference_dropout_layer_call_and_return_conditional_losses_6758l;в8
1в.
(К%
inputs         .. 
p
к "-в*
#К 
0         .. 
Ъ │
C__inference_dropout_2_layer_call_and_return_conditional_losses_6848l;в8
1в.
(К%
inputs         &&@
p
к "-в*
#К 
0         &&@
Ъ │
C__inference_dropout_2_layer_call_and_return_conditional_losses_6853l;в8
1в.
(К%
inputs         &&@
p 
к "-в*
#К 
0         &&@
Ъ ╫
B__inference_conv2d_2_layer_call_and_return_conditional_losses_6018Р89IвF
?в<
:К7
inputs+                           @
к "?в<
5К2
0+                           @
Ъ И
)__inference_activation_layer_call_fn_6738[7в4
-в*
(К%
inputs         .. 
к " К         .. ─
D__inference_sequential_layer_call_and_return_conditional_losses_6385|*+89JKTU^_EвB
;в8
.К+
conv2d_input         22
p

 
к "%в"
К
0         
Ъ │
C__inference_dropout_1_layer_call_and_return_conditional_losses_6808l;в8
1в.
(К%
inputs         **@
p 
к "-в*
#К 
0         **@
Ъ y
$__inference_dense_layer_call_fn_6891QJK1в.
'в$
"К
inputs         А╥
к "К         Й
&__inference_dropout_layer_call_fn_6773_;в8
1в.
(К%
inputs         .. 
p 
к " К         .. Й
&__inference_dropout_layer_call_fn_6768_;в8
1в.
(К%
inputs         .. 
p
к " К         .. Ц
)__inference_sequential_layer_call_fn_6711i*+89JKTU^_?в<
5в2
(К%
inputs         22
p

 
к "К         К
+__inference_activation_2_layer_call_fn_6828[7в4
-в*
(К%
inputs         &&@
к " К         &&@╫
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5994Р*+IвF
?в<
:К7
inputs+                            
к "?в<
5К2
0+                           @
Ъ п
'__inference_conv2d_1_layer_call_fn_6005Г*+IвF
?в<
:К7
inputs+                            
к "2К/+                           @╒
@__inference_conv2d_layer_call_and_return_conditional_losses_5970РIвF
?в<
:К7
inputs+                           
к "?в<
5К2
0+                            
Ъ Ц
)__inference_sequential_layer_call_fn_6728i*+89JKTU^_?в<
5в2
(К%
inputs         22
p 

 
к "К         в
F__inference_activation_5_layer_call_and_return_conditional_losses_6950X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ ░
D__inference_activation_layer_call_and_return_conditional_losses_6733h7в4
-в*
(К%
inputs         .. 
к "-в*
#К 
0         .. 
Ъ |
+__inference_activation_4_layer_call_fn_6928M0в-
&в#
!К
inputs         А
к "К         А
&__inference_flatten_layer_call_fn_6874U7в4
-в*
(К%
inputs         &&@
к "К         А╥д
F__inference_activation_4_layer_call_and_return_conditional_losses_6923Z0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ ╛
D__inference_sequential_layer_call_and_return_conditional_losses_6643v*+89JKTU^_?в<
5в2
(К%
inputs         22
p

 
к "%в"
К
0         
Ъ z
&__inference_dense_1_layer_call_fn_6918PTU/в,
%в"
 К
inputs         
к "К         АЛ
(__inference_dropout_2_layer_call_fn_6863_;в8
1в.
(К%
inputs         &&@
p 
к " К         &&@Л
(__inference_dropout_2_layer_call_fn_6858_;в8
1в.
(К%
inputs         &&@
p
к " К         &&@з
A__inference_flatten_layer_call_and_return_conditional_losses_6869b7в4
-в*
(К%
inputs         &&@
к "'в$
К
0         А╥
Ъ б
?__inference_dense_layer_call_and_return_conditional_losses_6884^JK1в.
'в$
"К
inputs         А╥
к "%в"
К
0         
Ъ в
A__inference_dense_2_layer_call_and_return_conditional_losses_6938]^_0в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ ▓
F__inference_activation_2_layer_call_and_return_conditional_losses_6823h7в4
-в*
(К%
inputs         &&@
к "-в*
#К 
0         &&@
Ъ н
%__inference_conv2d_layer_call_fn_5981ГIвF
?в<
:К7
inputs+                           
к "2К/+                            в
A__inference_dense_1_layer_call_and_return_conditional_losses_6911]TU/в,
%в"
 К
inputs         
к "&в#
К
0         А
Ъ ╛
D__inference_sequential_layer_call_and_return_conditional_losses_6694v*+89JKTU^_?в<
5в2
(К%
inputs         22
p 

 
к "%в"
К
0         
Ъ 