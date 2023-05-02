#####################################################
##########  선형회귀분석
##########  CH0607
#####################################################

#########################
library(MASS)
library(lmtest)


data(Boston)
head(Boston)
# 보스턴 집값 데이터 이 데이터는 보스턴 근교 지역의 집값 및 다른 정보를 포함한다.
# MASS 패키지를 설치하면 데이터를 로딩할 수 있다.
# 
# 
# B보스턴 근교 506개 지역에 대한 범죄율 (crim)등 14개의 변수로 구성
# • crim : 범죄율
# • zn: 25,000평방비트 기준 거지주 비율
# • indus: 비소매업종 점유 구역 비율
# • chas: 찰스강 인접 여부 (1=인접, 0=비인접)
# • nox: 일산화질소 농도 (천만개 당)
# • rm: 거주지의 평균 방 갯수 ***
# • age: 1940년 이전에 건축된 주택의 비율
# • dis: 보스턴 5대 사업지구와의 거리
# • rad: 고속도로 진입용이성 정도
# • tax: 재산세율 (10,000달러 당)
# • ptratio: 학생 대 교사 비율
# • black: 1000(B − 0.63)2, B: 아프리카계 미국인 비율
# • lstat : 저소득층 비율 ****
# • medv: 주택가격의 중앙값 (단위:1,000달러 당)


pairs(Boston[,which(names(Boston) %in% 
                      c('medv', 'rm', 'lstat'))], 
      pch=16, col='darkorange')

# pairs(Boston, pch=16, col='darkorange')
cor(Boston[,which(names(Boston) %in% 
                    c('medv', 'rm', 'lstat'))])

fit_Boston<-lm(medv~rm+lstat, data=Boston)
summary(fit_Boston)


## matrix

n = nrow(Boston)
X = cbind(rep(1,n), Boston$rm, Boston$lstat)
y = Boston$medv

beta_hat = solve(t(X)%*%X) %*% t(X) %*% y
beta_hat
coef(fit_Boston)

y_hat = X %*% beta_hat
y_hat[1:5]
fitted(fit_Boston)[1:5]

sse <- sum((y - y_hat)^2) ##SSE
sqrt(sse/(n-2-1)) ##RMSE
summary(fit_Boston)$sigma

##############################################################
dt <- Boston[,which(names(Boston) %in% c('medv', 'rm', 'lstat'))]
head(dt)

fit_Boston<-lm(medv~., data=dt)
fit_Boston<-lm(medv~rm+lstat, data=dt)

summary(fit_Boston)
## hat y = -1.3583 + 5.0948*rm - 0.6424*lstat


## 분산분석 : 회귀직선의 유의성 검정 
## H0 : beta1=beta2=0  vs.  H1:not H0

anova(fit_Boston) ## XXX

null_model <- lm(medv~1, data=dt)  #H0
fit_Boston <- lm(medv~., data=dt)  #H1

anova(null_model, fit_Boston) ##***

##beta의 신뢰구간 
confint(fit_Boston, level = 0.90)
coef(fit_Boston) + qt(0.975, 503) * summary(fit_Boston)$coef[,2]
coef(fit_Boston) - qt(0.975, 503) * summary(fit_Boston)$coef[,2]

vcov(fit_Boston)  ##var(hat beta) = (X^TX)^-1 \sigma^2

############# 평균반응, 개별 y 추정
## E(Y|x0), y = E(Y|x0) + epsilon
new_dt <- data.frame(rm=7, lstat=10)

# hat y0 = -1.3583 + 5.0948*7 - 0.6424*10
predict(fit_Boston, newdata = new_dt)
c(1,7,10)%*%beta_hat

predict(fit_Boston, 
        newdata = new_dt,
        interval = c("confidence"), 
        level = 0.95)  ##평균반응

predict(fit_Boston, newdata = new_dt, 
        interval = c("prediction"), 
        level = 0.95)  ## 개별 y

####
fit_Boston0 <- lm(medv ~ 0 + rm + lstat, dt)
summary(fit_Boston0)
summary(fit_Boston)


############ 잔차분석 
### epsilon : 선형성, 등분산성, 정규성, 독립성 

yhat <- fitted(fit_Boston)
res <- resid(fit_Boston)

plot(res ~ yhat,pch=16, ylab = 'Residual')
abline(h=0, lty=2, col='grey')


### 등분산성 
## H0 : 등분산  vs.  H1 : 이분산 (Heteroscedasticity)
bptest(fit_Boston)


## 잔차의 QQ plot
par(mfrow=c(1,2))
qqnorm(res, pch=16)
qqline(res, col = 2)

hist(res)
par(mfrow=c(1,1))

## Shapiro-Wilk Test
## H0 : normal distribution  vs. H1 : not H0
shapiro.test(res)


# 독립성검정 : DW test
dwtest(fit_Boston, alternative = "two.sided")  #H0 : uncorrelated vs H1 : rho != 0



############################################################
############################################################
###########  FM vs. RM

reduced_model = lm(medv ~ rm+lstat, data = Boston)
full_model = lm(medv ~ ., data=Boston)

summary(full_model)
summary(reduced_model)

anova(reduced_model, full_model)


# F = {(SSE_RM - SSE_FM)/r} / {SSE_FM/(n-p-1)}
p <- full_model$rank-1
q <- reduced_model$rank-1
SSE_FM <- anova(full_model)$Sum[p+1] #SSE_FM
SSE_RM <- anova(reduced_model)$Sum[q+1]  #SSE_RM

F0 <- ((SSE_RM-SSE_FM)/(p-q))/(SSE_FM/(nrow(Boston)-p-1))
F0

#기각역 F_{0.05}(p-q,n-p-1)
qf(0.95,p-q,nrow(Boston)-p-1)
# p-value
1-pf(F0, p-q,nrow(Boston)-p-1)


#################################
reduced_model = lm(medv ~ .-age-indus, data = Boston)
full_model = lm(medv ~ ., data=Boston)

anova(reduced_model, full_model)


############################################################
############################################################
##########  General linear hypothesis

x1<-c(4,8,9,8,8,12,6,10,6,9)
x2<-c(4,10,8,5,10,15,8,13,5,12)
y<-c(9,20,22,15,17,30,18,25,10,20)
fit<-lm(y~x1+x2)  ##FM
summary(fit)

## H0 : T*beta = c 

#H_0 : beta_1 = 1
linearHypothesis(fit, c(0,1,0), 1)

#b1-b2=0 => (0,1,-1) *beta 
#H_0 : beta_1 = beta2
linearHypothesis(fit, c(0,1,-1), 0)


#H_0 : beta_1 = beta2 + 1
linearHypothesis(fit, c(0,1,-1), 1)

#H_0 : beta_1 = beta2 + 5
linearHypothesis(fit, c(0,1,-1), 5)


##H_0 : beta_1 = beta2 + 1
#y=b0 + b1x1 + b2x2 + e = b0+x1 + b2(x1+x2)+e
#y-x1 = b0+b2(x1+x2)+e :   RM

y1 <- y-x1
z1 <- x1 + x2

fit2 <- lm(y1~z1)
summary(fit2)
anova(fit2)

anova(fit)  ##FM
anova(fit2)  #RM

# F = {(SSE_RM - SSE_FM)/r} / {SSE_FM/(n-p-1)}
p <- fit$rank-1
q <- fit2$rank-1
SSE_FM <- anova(fit)$Sum[p+1] #SSE_FM
SSE_RM <- anova(fit2)$Sum[q+1]  #SSE_RM

F0 <- ((SSE_RM-SSE_FM)/(p-q))/(SSE_FM/(length(y)-p-1))
F0

#기각역 F_{0.05}(p-q,n-p-1)
qf(0.95,p-q,length(y)-p-1)
# p-value
pf(F0, p-q,length(y)-p-1,lower.tail = F)


