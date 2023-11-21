include("init.jl")

X_data = gpu(rand(Float64,(1000,100)))


function l2_penalty(model::Flux.Chain)
    return sum([sum(abs2, layer.weight) for layer in model])
end


function l2_penalty_2(model::Flux.Chain)
    return sum(p -> sum(abs2, p), Flux.params(model))
end 

auto_encoder = gpu(Flux.Chain(
    Flux.Dense(100=>47), 
    Flux.Dense(47,2),
    Flux.Dense(2,47),
    Flux.Dense(47,100)));
l2_penalty_2(auto_encoder)
opt = Flux.ADAM(1e-3)
function train(auto_encoder, X_data)
    for i in 1:100
        out = auto_encoder(X_data')
        ps = Flux.params(auto_encoder)
        gs = gradient(ps) do 
            lossval = Flux.Losses.mse(out', X_data) + l2_penalty(auto_encoder)
        end 
        println("$i - $(Flux.Losses.mse(out', X_data) + l2_penalty(auto_encoder))")
        Flux.update!(opt, ps, gs)
    end 
end 

function train_2(auto_encoder, X_data)
    for i in 1:100
        out = auto_encoder(X_data')
        ps = Flux.params(auto_encoder)
        gs = gradient(ps) do 
            lossval = Flux.Losses.mse(out', X_data) + l2_penalty_2(auto_encoder)
        end 
        println("$i - $(Flux.Losses.mse(out', X_data) + l2_penalty_2(auto_encoder))")
        Flux.update!(opt, ps, gs)
    end 
end 

train_2(auto_encoder, X_data)

[size(layer.weight) for layer in auto_encoder]