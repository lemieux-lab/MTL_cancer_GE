include("init.jl")

X_data = gpu(rand(Float64,(100,1000)))
ps = Flux.params(auto_encoder)
function l2_penalty(ps)
    return sum(p -> sum(abs2, p), ps)
end 

auto_encoder = gpu(Flux.Chain(
    Flux.Dense(100=>47), 
    Flux.Dense(47,2),
    Flux.Dense(2,47),
    Flux.Dense(47,100)));

opt = Flux.ADAM(1e-3)

ps[5]

function lossf(model, X, ps)
    return Flux.Losses.mse(vec(model(X)), vec(X)) + l2_penalty(ps)
end

function train(auto_encoder, X_data)
    for i in 1:100
        gs = gradient(ps) do 
            lossval = lossf(auto_encoder, X_data, ps ) #+ l2_penalty_2(auto_encoder)
        end 
        println("$i - $(lossf(auto_encoder, X_data, ps ))")#+ l2_penalty_2(auto_encoder))")
        Flux.update!(opt, ps, gs)
    end 
end 

train(auto_encoder, X_data)
