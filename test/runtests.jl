println("Starting runtests.jl ...")

using ExtremelyRandomizedTrees
using FactCheck


# function unittest()
# 	for i = 1:101
# 		data = single(rand(1,100000)<(i-1)/100) + 1;
# 		Hs(i) = entropy(data);
# 	end
# end

# function unittest2()
#     x = ceil(eps+10*rand(1,100))
#     y = x;
#     p.rteNClasses = max(x);
    
#     for i = 1:100;
#         mI(i) = mutualInformation(p,x,y);
#         y(i) = 1;
#     end
# end

println("  ... runtests.jl done!")
