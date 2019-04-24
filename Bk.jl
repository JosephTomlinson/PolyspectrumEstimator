@everywhere const np = nworkers()
@everywhere using DistributedArrays
@everywhere using DistributedArrays.SPMD
@everywhere using FFTW
@everywhere using LinearAlgebra
@everywhere using ParallelDataTransfer

@everywhere nkmax = 21
@everywhere skip = 1


@everywhere function transpose3D(x)
    permutedims(x,(2,1,3))
end


@everywhere @views function alltoalltranspose3D(x, nx, recievearray; pids=workers())
    pstart = minimum(pids)
    np = length(pids)
    rank = myid() - pstart + 1
    ny = size(x,2)
    extrax = nx % np
    extray = ny % np
    if rank <= extrax
        haveextra = true
    else
        haveextra = false
    end
    if rank <= extray
        haveextray = true
    else
        haveextray = false
    end

    if haveextra
        dx = size(x,1) - 1
    else
        dx = size(x,1)
    end
    nz = size(x,3)
    dy = convert(Int64,floor(ny/np))


    xt = transpose3D(x)

    i=1
    while i < np
        i *= 2
    end
    if i == np
        pow2 = true
    else
        pow2 = false
    end

    if haveextra
        if haveextray
            offset = rank - 1
            recievearray[:,(1+dx*(rank-1)+offset):dx*rank+(offset+1),:] .= 
                        xt[(1+dy*(rank-1)+offset):dy*rank+(offset+1),:,:] 
        else
            offsetx = rank - 1
            offsety = extray
            recievearray[:,(1+dx*(rank-1)+offsetx):dx*rank+(offsetx+1),:] .= 
                        xt[(1+dy*(rank-1)+offsety):dy*rank+(offsety),:,:] 
        end
    else
        if haveextray
            offsetx = extrax
            offsety = rank - 1
            recievearray[:,(1+dx*(rank-1)+offsetx):dx*rank+(offsetx),:] .= 
                        xt[(1+dy*(rank-1)+offsety):dy*rank+(offsety+1),:,:] 
        else
            offsetx = extrax
            offsety = extray
            recievearray[:,(1+dx*(rank-1)+offsetx):dx*rank+offsetx,:] .= 
                        xt[(1+dy*(rank-1)+offsety):dy*rank+offsety,:,:] 
        end
    end

    for i = 1:(np-1)
        if pow2
            source = destination = ((rank-1) ⊻ i) + 1
        else
            source = (rank - i - 1 + np ) % np + 1
            destination = (rank + i - 1 + np) % np + 1
        end
        if destination <= extray
            offset = destination - 1
            SPMD.sendto(destination + pstart - 1, xt[(1+dy*(destination-1)+offset):dy*destination+(offset+1),:,:])
        else
            offset = extray
            SPMD.sendto(destination + pstart - 1, xt[(1+dy*(destination-1)+offset):dy*destination+offset,:,:])
        end

        if source <= extrax
            offset = source - 1
            recievearray[:,(1+dx*(source-1)+offset):dx*source+(offset+1),:] .= SPMD.recvfrom(source + pstart - 1)
        else
            offset = extrax
            recievearray[:,(1+dx*(source-1)+offset):dx*source+offset,:] .= SPMD.recvfrom(source + pstart - 1)
        end
        barrier(; pids=pids)
    end
end

@everywhere function generate_plans(delkix,z)

    plan1 = plan_bfft!(delkix[:L], (2,3), flags=FFTW.MEASURE)
    plan2 = plan_brfft(z, size(z,3), (2,), flags=FFTW.MEASURE)
    return [plan1, plan2]
end

@everywhere function apply_plans(dk,dr,plans,z)
    pids = procs(dr)[:,1]
    plans[1]*dk[:L]
    alltoalltranspose3D(dk[:L], size(dk,1), z; pids=pids)
    mul!(dr[:L], plans[2], z)
    #w = plans[2]*z
    #alltoalltranspose3D(w, size(dk,2), dr[:L]; pids=pids)
end


@everywhere function delta_ki_fft!(dk,delkix,dr,delkir)
    pids = procs(dk)
    pstart = minimum(pids)
    bymesh = size(dr,1)
    bxmesh = size(dr,2)
    bzmesh = size(dr,3)
    bcnmesh = convert(Int64,bxmesh/2) + 1

    haveextra = ((myid() - pstart + 1) <= rem(size(delkix,2), np))
    if haveextra
        transpose_size = (div(bymesh, np) + 1, bcnmesh, bzmesh)
    else
        transpose_size = (div(bymesh, np), bcnmesh, bzmesh)
    end

    z = zeros(Complex{Float64}, transpose_size)

    plans = generate_plans(dk,z)

    for nkindx = 1:nkmax
        @inbounds broadcast!(identity, dk[:L], view(delkix[:L],:,:,:,nkindx))   
        apply_plans(dk,dr,plans,z)
        @inbounds broadcast!(identity, view(delkir[:L],:,:,:,nkindx), dr[:L])
    end
end

@everywhere function compute_pk_bk!(bk,delkir,dr)
    pids = procs(dr)[:,1]
    pstart = minimum(pids)
    bymesh = size(dr,1)
    bxmesh = size(dr,2)
    bzmesh = size(dr,3)
    for j1 = 1:nkmax, j2 = 1:j1, j3 = (j1-j2):j2
        if j3 > 0
            @inbounds broadcast!(*, dr[:L], view(delkir[:L],:,:,:,j1),
                view(delkir[:L],:,:,:,j2), view(delkir[:L],:,:,:,j3))
            barrier(;pids=pids)
            if myid() == pstart
                bk[j1,j2,j3] = sum(dr)/(bxmesh*bymesh*bzmesh)
            end
            barrier(;pids=pids)
        end
    end
end

function run_Bk(bnmesh, nkmax, skip)
    bcnmesh = convert(Int64,bnmesh/2) + 1
    ndkmax = skip*nkmax

    @everywhere pids = workers()
    @everywhere pstart = minimum(pids)

    bk = Array{Float64,3}(undef,nkmax,nkmax,nkmax)
    @defineat pstart bk = Array{Float64,3}(undef,nkmax,nkmax,nkmax)

    delkix = dzeros(Complex{Float64}, (bcnmesh, bnmesh, bnmesh, nkmax), pids, [np,1,1,1])
    dk = dzeros(Complex{Float64}, (bcnmesh, bnmesh, bnmesh), pids, [np,1,1])
    delkir = dzeros((bnmesh,bnmesh,bnmesh,nkmax), pids, [np,1,1,1])
    dr = dzeros((bnmesh,bnmesh,bnmesh), pids, [np,1,1]);
    
    @everywhere function fourier_space_density!(delkix::DArray)
        bnmesh = size(delkix,2)
        bcnmesh = size(delkix,1)
        nkmax = size(delkix,4)
        for k = 1:(bnmesh)
            kz = k - 1
            if (k-1) >= bcnmesh
                kz = kz - bnmesh
            end
            for j = 1:(bnmesh)
                ky = j - 1
                if (j-1) >= bcnmesh
                    ky = ky - bnmesh
                end
                #for i = 1:(bcnmesh+1)
                for i = 1:length(localindices(delkix)[1])
                    kx = i - 1 + localindices(delkix)[1][1] - 1
                    magnitude = sqrt(kx^2+ky^2+kz^2)
                    numnk = convert(Int64,round(magnitude/skip))

                    if numnk <= nkmax && numnk > 0
                        delkix[:L][i,j,k,numnk] = 1.0 + 0.0im
                    end
                end
            end
        end
    end
    spmd(fourier_space_density!, delkix; pids=pids)
    @spawnat pstart fill!(delkix[:L][1,1,1,:],0.0 + 0.0im);
    
    spmd(delta_ki_fft!, dk, delkix, dr, delkir; pids=pids)
    
    spmd(compute_pk_bk!, bk, delkir, dr; pids=pids)
    
    d_closeall()    
end
