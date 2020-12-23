KDE_test <- function(X1, Y1, X2, Y2){
	if(length(dim(X1)) == 1){
		X1 <- as.vector(X1)
		X2 <- as.vector(X2)
		Y1 <- as.vector(Y1)
		Y2 <- as.vector(Y2)
		res_time <- kde.test(x1=X1, x2=X2)$pvalue
		res_ampl <- kde.test(x1=Y1, x2=Y2)$pvalue
		mat1 <- matrix(c(X1, Y1), nrow=length(X1))
		mat2 <- matrix(c(X2, Y2), nrow=length(X2))
		res_2d <- kde.test(x1=mat1, x2=mat2)$pvalue	
		return(c(res_time, res_ampl, res_2d))
	}
	results <- matrix(, nrow = nrow(X1) * nrow(X2), ncol = 3)
	index <- 1
	#
	for(i1 in 1:nrow(X1)) {
		#
		x1 <- X1[i1, ]
		x1 <- x1[x1 >= 0]
		y1 <- Y1[i1, ]
		y1 <- y1[y1 >= 0]
		#
		for(i2 in 1:nrow(X2)) {
			#
			x2 <- X2[i2, ]
			x2 <- x2[x2 >= 0]
			y2 <- Y2[i2, ]
			y2 <- y2[y2 >= 0]
			#
			mat1 <- matrix(c(x1, y1), nrow=length(x1))
			mat2 <- matrix(c(x2, y2), nrow=length(x2))
			#
			res_time <- kde.test(x1=x1, x2=x2)$pvalue
			res_ampl <- kde.test(x1=y1, x2=y2)$pvalue
			res_2d <- kde.test(x1=mat1, x2=mat2)$pvalue

			results[index, ] <- c(res_time, res_ampl, res_2d)
			index <- index + 1
		} 
	}
	return(results)
}